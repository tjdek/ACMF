""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from typing import Tuple
@torch.no_grad()
def _covariance_map(feat: torch.Tensor) -> torch.Tensor:
    """Eq.(4): Σ_i = (1/HW) * W_i W_i^T. feat: [B,C,H,W] -> [B,C,C]"""
    if feat.dim() != 4:
        raise ValueError(f"feat must be [B,C,H,W], got {feat.shape}")
    B, C, H, W = feat.shape
    x = feat.reshape(B, C, H * W)                         # [B, C, HW]
    sigma = torch.bmm(x, x.transpose(1, 2)) / (H * W)     # [B, C, C]
    return sigma


@torch.no_grad()
def compute_V_and_S_from_concat(feat: torch.Tensor,feat_aug: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    sigma = _covariance_map(feat)         # [B,C,C]
    sigma_aug = _covariance_map(feat_aug) # [B,C,C]

    mu = 0.5 * (sigma + sigma_aug)
    per_sample_var = 0.5 * ((sigma - mu) ** 2 + (sigma_aug - mu) ** 2)  # [B,C,C]
    V = per_sample_var.mean(dim=0)  # [C,C]
    S = V.mean(dim=1)               # [C]
    total_sum = torch.sum(S)
    # print("total_sum",total_sum)
    # Sen=torch.abs(sigma - sigma_aug).mean(dim=0).mean(dim=1)
    # print("Sen",total_sum)
    return V, S, total_sum

@torch.no_grad()
def get_idx_from_x5(x5: torch.Tensor,x5_argu: torch.Tensor, mode: str = "sen",ratio: float = 0.1):
    """
    x5_concat: [2B,C,H,W]
    return: low_idx [K]
    """
    _, S,total_sum = compute_V_and_S_from_concat(x5,x5_argu)  # S: [C]
    C = S.numel()
    if mode == "sen":
        K = max(1, int(round(ratio * C)))
        idx = torch.topk(S, k=K, largest=False).indices  
    elif mode == "insen":
        K = max(1, int(round((ratio) * C)))
        idx = torch.topk(S, k=K, largest=True).indices  
    remaining_mask = torch.ones(C, dtype=torch.bool, device=x5.device)
    remaining_mask[idx] = False

    return idx,total_sum

def get_feature(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if idx is None:
        return x
    if idx.numel() == 0:
        return x
    if idx.dtype != torch.long:
        idx = idx.long()
    idx = idx.to(x.device)

    out = x.clone()                   # out[:, idx, :, :] = 0.0
    B, C, H, W = x.shape

    fill = x.mean(dim=1, keepdim=True)  # [B,1,H,W]
    fill_k = fill.expand(B, idx.numel(), H, W)  # [B,K,H,W]

    out[:, idx, :, :] = fill_k

    return out

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,pos_basic_dims=1024,):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.pos_basic_dims=pos_basic_dims
        # self.channel_mask = AdaptiveChannelMask(mask_percent= 0.25,mask_value= "zero")
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def encode(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5, (x4, x3, x2, x1)

    def decoder(self, x5_, x4_, x3_, x2_, x1_):
        x11 = self.up1(x5_, x4_)
        x22 = self.up2(x11, x3_)
        x33 = self.up3(x22, x2_)
        x44 = self.up4(x33, x1_)
        return self.outc(x44),(x44, x33, x22, x11)

    def decode(self, x5, skips):
        x4, x3, x2, x1 = skips
        out,d_f=self.decoder(x5, x4, x3, x2, x1)
        return out,d_f

    def forward(self,x: torch.Tensor):
        x5, skips_lb = self.encode(x)  
        x4, x3, x2, x1 = skips_lb
        logits,d_f=self.decode(x5,skips_lb)
        x44, x33, x22, x11=d_f
        # print("x5,shape",x5.shape)

        return logits,x5

    def forward_pair(self, x_lb, x_ulb,x_lb_trans,x_ulb_trans):
        x5_lb, skips_lb = self.encode(x_lb)  
        x5_ulb, skips_ulb = self.encode(x_ulb)  

        x5_lb_trans, skips_lb_lb_trans = self.encode(x_lb_trans)
        x5_ulb_trans, skips_ulb_ulb_trans = self.encode(x_ulb_trans)
        #
        idx_L,total_sum_L=get_idx_from_x5(x5_lb,x5_lb_trans)
        idx_U,total_sum_U=get_idx_from_x5(x5_ulb,x5_ulb_trans)
        idx_L = idx_L.view(-1).long()
        idx_U = idx_U.view(-1).long().to(idx_L.device)
        # # idx_L ∩ idx_U
        idx_common = idx_L[torch.isin(idx_L, idx_U)]
        x5_lb_mask=get_feature(x5_lb,idx_common)
        x5_ulb_mask=get_feature(x5_ulb,idx_common)

        logit_lb,_ = self.decode(x5_lb_mask, skips_lb)
        logit_ulb,_ = self.decode(x5_ulb_mask, skips_ulb)
        logit_lb_trans,_ =self.decode(x5_lb_trans,skips_lb_lb_trans)
        logit_ulb_trans,_=self.decode(x5_ulb_trans,skips_ulb_ulb_trans)

        # logit_lb ,_= self.decode(x5_lb, skips_lb)
        # logit_ulb,_ = self.decode(x5_ulb, skips_ulb)
        # logit_lb_trans,_ =self.decode(x5_lb_trans,skips_lb_lb_trans)
        # logit_ulb_trans,_=self.decode(x5_ulb_trans,skips_ulb_ulb_trans)

        return logit_lb, logit_ulb,logit_lb_trans,logit_ulb_trans,total_sum_L,total_sum_U




