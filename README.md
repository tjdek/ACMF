# Adaptive-Channel-Masking-Framework
When processing multi-domain datasets from multiple medical centers, the per formance of traditional SSL is often limited by domain shift, which prevents it from providing reliable pseudo-labels for unlabeled data.  We propose a channel-masking framework to address the issue of low-quality pseudo labels. For the concept of channel-sensitivity, please refer to [Domaindropout](https://github.com/lingeringlight/DomainDrop).
## Ablation Study on Fundus Dataset (5% Labeled Data)
To evaluate the performance of ACM in the scenario with lower labeled data (5%), we further analyzed the collaborative contributions of different components (**DSS**, **CS**, and **SCM**) in four target domains, and used the *Dice* as the evaluation metric. We will present more results in our subsequent journal papers.
| CS | DSS | SCM | 🟩Lable: Domain 1 | 🟦Lable: Domain 2 | 🟨Lable: Domain 3 | 🟪Lable: Domain 4 | 🔴Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ❌ | ❌ | ❌ | 🟩0.6220 | 🟦0.6817 | 🟨0.6910 | 🟪0.5893 |🔴0.6560 |
| ✔️ | ❌ | ❌ | 🟩**0.6861** | 🟦0.6651 | 🟨0.7501 | 🟪0.6043 | 🔴0.6764 |
| ✔️ | ✔️ | ❌ | 🟩0.6839 | 🟦0.6795 | 🟨0.7433 | 🟪0.6137 | 🔴0.6801 |
| **✔️** | **✔️** | **✔️** | 🟩0.6524 | 🟦**0.6889** | 🟨**0.7701** | 🟪**0.6797** | 🔴**0.6978** |

*Note*: 🟩 **Label: Domain 1** *indicates that Domain 1 is utilized as the labeled source domain during training, and* 🟩0.6220 *represent the average performance evaluated across all domains. 
**Bold** values indicate the optimal setting or the highest performance in each column.*
### Data
**Fundus**: https://github.com/emma-sjwang/Dofe <br>
**M&M**: https://www.kaggle.com/datasets/tailength/m-and-m2-dataset 
### Train
```text
python train_Fundus.py --dataset fundus --lb_domain 4 --lb_num 5 --save_name *** --gpu 0
```
### Test
```text
python test.py --dataset fundus --lb_domain 4 --lb_num 5 --save_name *** --gpu 0
```
---
