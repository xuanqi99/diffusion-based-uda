# Diffusion-UDA Research Workspace

This directory is a research workspace for studying how diffusion-based
generative models can improve unsupervised domain adaptation (UDA). The current
implementation provides runnable PyTorch ERM, DANN, AFN, CDAN, MDD, JAN, CAN,
GTA, ADDA, MCD, SymmNets, GVB-GD, ETD, SRDC, ACTIR, TCM, ICDA, iMSDA, UniOT, WDGRL, and PPOT baselines, which are useful as clean reference points before adding
diffusion-generated images, diffusion-based feature regularization, or other
generative adaptation strategies.

ERM trains only on labeled source-domain images and evaluates on the target
domain when target labels are available. This makes it a practical baseline for
measuring whether future diffusion-assisted UDA components provide real gains.
DANN trains with source classification loss plus a domain-adversarial loss over
source and target images, while never using target class labels for training.
AFN trains with source classification loss plus adaptive feature norm
regularization over source and target images, while never using target class
labels for training. CDAN trains with source classification loss plus a
conditional domain-adversarial loss whose discriminator sees the multilinear
interaction between features and classifier predictions. It supports the common
randomized multilinear map for large feature/class spaces, exact multilinear
conditioning, and optional CDAN+E entropy conditioning. MDD trains with source
classification loss plus margin disparity discrepancy from a second adversarial
classifier head, aligning domains without using target labels; this follows the
MDD baseline reported in `A Closer Look at Smoothness in Domain Adversarial
Training`. JAN trains with source classification loss plus joint maximum mean
discrepancy over source and target feature-prediction distributions, following
`Deep Transfer Learning with Joint Adaptation Networks`. CAN trains with source
classification loss plus class-aware contrastive domain discrepancy over source
labels and target pseudo-labels, following `Contrastive Adaptation Network for
Unsupervised Domain Adaptation`. GTA trains with source
classification loss plus an auxiliary-classifier GAN branch whose generator and
discriminator provide adaptation gradients for source and unlabeled target
embeddings, following `Generate To Adapt`. ADDA first pretrains a source encoder
and classifier, then freezes them while adversarially training a separate target
encoder to fool a domain discriminator, following `Adversarial Discriminative
Domain Adaptation`. MCD trains a shared feature extractor with two classifier
heads, maximizes their target prediction discrepancy while preserving source
classification, then updates the feature extractor to minimize that discrepancy,
following `Maximum Classifier Discrepancy for Unsupervised Domain Adaptation`.
SymmNets trains symmetric source and target task classifiers, uses their
concatenated logits as a shared 2K classifier for domain discrimination, and
updates the feature extractor with category-level and domain-level confusion,
following `Domain-Symmetric Networks for Adversarial Domain Adaptation`.
GVB-GD trains a source classifier with a gradually vanishing generator bridge,
feeds classifier probabilities to an entropy-weighted domain discriminator, and
uses a discriminator bridge to reduce over-confident adversarial alignment,
following `Gradually Vanishing Bridge for Adversarial Domain Adaptation`.
ETD trains with source classification loss plus an attention-reweighted
transport distance between source and target feature batches. It represents the
Kantorovich potentials with neural networks and adds target entropy
minimization, following `Enhanced Transport Distance for Unsupervised Domain
Adaptation`.
SRDC avoids explicit domain alignment and instead uncovers target-domain
discrimination through deep clustering. It combines target auxiliary-distribution
clustering, feature-space clustering, source structural regularization, and
optional source sample soft selection, following `Unsupervised Domain Adaptation
via Structurally Regularized Deep Clustering`.
ACTIR learns an invariant classifier component and a transportable adaptive
component. This UDA implementation trains source labels with combined and
invariant losses, penalizes conditional dependence between the two components,
adds an adaptive-gradient penalty, and adapts on confident target pseudo-labels,
following `Invariant and Transportable Representations for Anti-Causal Domain
Shifts`.
TCM transports causal mechanisms by learning multiple proxy mechanism heads and
making interventional predictions over source/target mechanism priors. This
lightweight feature-level implementation aligns mechanism proxies, regularizes
mechanism diversity, and adapts on confident target pseudo-labels, following
`Transporting Causal Mechanisms for Unsupervised Domain Adaptation`.
ICDA performs implicit class-conditioned domain alignment. It uses target
pseudo-labels only to choose class-compatible source and target samples for the
domain-adversarial loss, avoiding direct target pseudo-label supervision while
reducing class-misaligned domain discriminator shortcuts, following `Implicit
Class-Conditioned Domain Alignment for Unsupervised Domain Adaptation`.
iMSDA learns a partially identifiable latent representation with invariant and
changing subspaces. This feature-level implementation uses a VAE-style encoder
and decoder over backbone features, a domain-specific invertible affine flow for
the changing subspace, source classification, and target entropy minimization,
following `Partial disentanglement for domain adaptation`.
UniOT uses optimal transport to jointly route target samples to source-class
prototypes and target-private prototypes. This feature-level implementation
uses adaptive common/private mass filling from target prediction statistics,
Sinkhorn transport, source prototype compactness, target common-class soft
training, and private-prototype clustering, following `Unified Optimal Transport
Framework for Universal Domain Adaptation`.
WDGRL trains a domain critic to estimate the empirical Wasserstein distance
between source and target feature distributions. The critic is regularized with
a gradient penalty, while the feature extractor minimizes source classification
loss plus the critic-estimated Wasserstein distance, following `Wasserstein
Distance Guided Representation Learning for Domain Adaptation`.
PPOT learns a source-target transport plan whose probabilities are polarized
toward explicit intra-class and inter-class structure. This mini-batch
implementation combines feature and semantic OT costs, Sinkhorn transport,
dynamic probability-polarization margins from source labels and target
pseudo-labels, source classification, and transported-source supervision,
following `Probability-Polarized Optimal Transport for Unsupervised Domain
Adaptation`.

Supported dataset presets:

- `officehome`: Art, Clipart, Product, Real_World
- `office31`: amazon, dslr, webcam
- `visda2017`: train/synthetic source and validation/real target
- `domainnet`: clipart, infograph, painting, quickdraw, real, sketch

## Install

```powershell
pip install -r uda/requirements.txt
```

## Expected ImageFolder Layout

The loader accepts either a global data root or the dataset root itself. These
layouts both work:

```text
data/
  officehome/
    Art/
      class_a/image_001.jpg
    Clipart/
      class_a/image_002.jpg
```

```text
OfficeHome/
  Art/
    class_a/image_001.jpg
  Clipart/
    class_a/image_002.jpg
```

## Methods

- `ERM`: source-only empirical risk minimization.
- `DANN`: domain-adversarial neural network with a gradient reversal layer.
- `AFN`: adaptive feature norm, with `SAFN` by default and optional `HAFN`.
- `CDAN`: conditional domain-adversarial network with randomized multilinear
  conditioning by default and optional `--entropy-conditioning` for CDAN+E.
- `MDD`: margin disparity discrepancy with a main classifier head and an
  adversarial classifier head, following the MDD baseline reported by Rangwani
  et al. in `A Closer Look at Smoothness in Domain Adversarial Training`.
- `JAN`: joint adaptation network with multi-kernel JMMD over features and
  classifier predictions.
- `CAN`: contrastive adaptation network with class-aware CDD, target
  pseudo-labeling, intra-class domain alignment, and inter-class separation.
- `GTA`: generate-to-adapt training with an F-C classification stream and an
  auxiliary-classifier GAN stream over generated source-like images.
- `ADDA`: adversarial discriminative domain adaptation with separate source and
  target encoders trained in source-supervised and target-adversarial stages.
- `MCD`: maximum classifier discrepancy with two task classifiers that expose
  and then reduce target-domain prediction disagreement.
- `SymmNets`: domain-symmetric networks with source and target task classifiers,
  a shared 2K classifier, category-level confusion, domain-level confusion, and
  target entropy minimization.
- `GVB-GD`: gradually vanishing bridge for the classifier generator and domain
  discriminator with entropy-weighted adversarial alignment.
- `ETD`: enhanced transport distance with attention-aware transport costs,
  neural Kantorovich potentials, and target entropy minimization.
- `SRDC`: structurally regularized deep clustering with target
  auxiliary-distribution clustering, feature-space Student-t clustering, source
  structural regularization, and optional soft source sample weighting.
- `ACTIR`: adaptive-invariant representation learning with invariant and
  adaptive classifier heads, conditional decorrelation, an adaptive-gradient
  penalty, and target pseudo-label adaptation.
- `TCM`: transporting causal mechanisms with learned mechanism proxies,
  interventional mechanism-prior prediction, mechanism alignment, diversity
  regularization, and confident target pseudo-label adaptation.
- `ICDA`: implicit class-conditioned domain alignment with pseudo-label-guided
  source/target sample selection for domain-adversarial training.
- `iMSDA`: identifiable multi-source domain adaptation style latent learning
  with invariant/changing latent partitioning, domain-specific invertible
  changing-part normalization, source classification, target entropy, and
  feature-level VAE regularization.
- `UniOT`: unified optimal transport with source-class prototypes, learnable
  target-private prototypes, adaptive common/private mass filling, Sinkhorn
  transport, and target representation learning.
- `WDGRL`: Wasserstein distance guided representation learning with a neural
  domain critic, WGAN-GP-style gradient penalty, source classification, and
  feature-level Wasserstein alignment.
- `PPOT`: probability-polarized optimal transport with feature/semantic costs,
  dynamic intra/inter-class transport margins, target pseudo-label structure,
  and transported-source target-space supervision.

## Run Examples

OfficeHome:

```powershell
python uda/erm.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with DANN:

```powershell
python uda/dann.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with AFN:

```powershell
python uda/afn.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with CDAN:

```powershell
python uda/cdan.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with CDAN+E entropy conditioning:

```powershell
python uda/cdan.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32 --entropy-conditioning
```

OfficeHome with MDD:

```powershell
python uda/mdd.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with JAN:

```powershell
python uda/jan.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with CAN:

```powershell
python uda/can.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with GTA:

```powershell
python uda/gta.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with ADDA:

```powershell
python uda/adda.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --source-epochs 5 --epochs 20 --batch-size 32
```

OfficeHome with MCD:

```powershell
python uda/mcd.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with SymmNets:

```powershell
python uda/symmnets.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with GVB-GD:

```powershell
python uda/gvbgd.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with ETD:

```powershell
python uda/etd.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32 --optimizer adamw
```

OfficeHome with SRDC:

```powershell
python uda/srdc.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with ACTIR:

```powershell
python uda/actir.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with TCM:

```powershell
python uda/tcm.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with ICDA:

```powershell
python uda/icda.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with iMSDA:

```powershell
python uda/imsda.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with UniOT:

```powershell
python uda/uniot.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with WDGRL:

```powershell
python uda/wdgrl.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with PPOT:

```powershell
python uda/ppot.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

Office31:

```powershell
python uda/erm.py --data-root D:\datasets --dataset office31 --source amazon --target webcam --arch resnet50 --epochs 20
```

VisDA2017:

```powershell
python uda/erm.py --data-root D:\datasets --dataset visda2017 --source train --target validation --arch resnet50 --epochs 20
```

DomainNet:

```powershell
python uda/erm.py --data-root D:\datasets --dataset domainnet --source real --target sketch --arch resnet50 --epochs 20
```

## List File Mode

For custom splits, pass list files. Source labels are required. Target labels are
optional; unlabeled target rows are skipped during metric calculation.

```text
relative/path/image_001.jpg 0
relative/path/image_002.jpg 1
```

```powershell
python uda/erm.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with DANN:

```powershell
python uda/dann.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with AFN:

```powershell
python uda/afn.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with CDAN:

```powershell
python uda/cdan.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with MDD:

```powershell
python uda/mdd.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with JAN:

```powershell
python uda/jan.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with CAN:

```powershell
python uda/can.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with GTA:

```powershell
python uda/gta.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with ADDA:

```powershell
python uda/adda.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with MCD:

```powershell
python uda/mcd.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with SymmNets:

```powershell
python uda/symmnets.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with GVB-GD:

```powershell
python uda/gvbgd.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with ETD:

```powershell
python uda/etd.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65 --optimizer adamw
```

The same split files can be used with SRDC:

```powershell
python uda/srdc.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with ACTIR:

```powershell
python uda/actir.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with TCM:

```powershell
python uda/tcm.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with ICDA:

```powershell
python uda/icda.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with iMSDA:

```powershell
python uda/imsda.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with UniOT:

```powershell
python uda/uniot.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with WDGRL:

```powershell
python uda/wdgrl.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

The same split files can be used with PPOT:

```powershell
python uda/ppot.py --data-root D:\datasets --dataset officehome --source-list source.txt --target-list target.txt --num-classes 65
```

Outputs are written to `runs/` by default and include `config.json`,
`metrics.csv`, `checkpoint_last.pt`, and `best_target.pt` when target labels are
available.

## References

- Guoliang Kang, Lu Jiang, Yi Yang, and Alexander G. Hauptmann. `Contrastive
  Adaptation Network for Unsupervised Domain Adaptation`, CVPR 2019.
  https://openaccess.thecvf.com/content_CVPR_2019/papers/Kang_Contrastive_Adaptation_Network_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf
- Shuhao Cui, Shuhui Wang, Junbao Zhuo, Chi Su, Qingming Huang, and Qi Tian.
  `Gradually Vanishing Bridge for Adversarial Domain Adaptation`, CVPR 2020.
  https://openaccess.thecvf.com/content_CVPR_2020/html/Cui_Gradually_Vanishing_Bridge_for_Adversarial_Domain_Adaptation_CVPR_2020_paper.html
- Yabin Zhang, Hui Tang, Kui Jia, and Mingkui Tan. `Domain-Symmetric Networks
  for Adversarial Domain Adaptation`, CVPR 2019.
  https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Domain-Symmetric_Networks_for_Adversarial_Domain_Adaptation_CVPR_2019_paper.html
- Kuniaki Saito, Kohei Watanabe, Yoshitaka Ushiku, and Tatsuya Harada.
  `Maximum Classifier Discrepancy for Unsupervised Domain Adaptation`, CVPR
  2018.
  https://openaccess.thecvf.com/content_cvpr_2018/html/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.html
- Eric Tzeng, Judy Hoffman, Kate Saenko, and Trevor Darrell. `Adversarial
  Discriminative Domain Adaptation`, CVPR 2017.
  https://openaccess.thecvf.com/content_cvpr_2017/html/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.html
- Swami Sankaranarayanan, Yogesh Balaji, Carlos D. Castillo, and Rama Chellappa.
  `Generate To Adapt: Aligning Domains using Generative Adversarial Networks`,
  CVPR 2018.
  https://openaccess.thecvf.com/content_cvpr_2018/papers/Sankaranarayanan_Generate_to_Adapt_CVPR_2018_paper.pdf
- Mingsheng Long, Han Zhu, Jianmin Wang, and Michael I. Jordan. `Deep Transfer
  Learning with Joint Adaptation Networks`, ICML 2017.
  https://proceedings.mlr.press/v70/long17a.html
- Harsh Rangwani, Sumukh K Aithal, Mayank Mishra, Arihant Jain, and R.
  Venkatesh Babu. `A Closer Look at Smoothness in Domain Adversarial Training`,
  ICML 2022. https://proceedings.mlr.press/v162/rangwani22a.html
- Mengxue Li, Yi-Ming Zhai, You-Wei Luo, Peng-Fei Ge, and Chuan-Xian Ren.
  `Enhanced Transport Distance for Unsupervised Domain Adaptation`, CVPR 2020.
  https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Enhanced_Transport_Distance_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.html
- Hui Tang, Ke Chen, and Kui Jia. `Unsupervised Domain Adaptation via
  Structurally Regularized Deep Clustering`, CVPR 2020.
  https://openaccess.thecvf.com/content_CVPR_2020/html/Tang_Unsupervised_Domain_Adaptation_via_Structurally_Regularized_Deep_Clustering_CVPR_2020_paper.html
- Yibo Jiang and Victor Veitch. `Invariant and Transportable Representations for
  Anti-Causal Domain Shifts`, NeurIPS 2022 Workshop on Distribution Shifts.
  https://openreview.net/forum?id=w1FmeUdwxEg
- Zhongqi Yue, Tan Wang, Qianru Sun, Xian-Sheng Hua, and Hanwang Zhang.
  `Transporting Causal Mechanisms for Unsupervised Domain Adaptation`, ICCV
  2021.
  https://openaccess.thecvf.com/content/ICCV2021/html/Yue_Transporting_Causal_Mechanisms_for_Unsupervised_Domain_Adaptation_ICCV_2021_paper.html
- Xiang Jiang, Qicheng Lao, Stan Matwin, and Mohammad Havaei. `Implicit
  Class-Conditioned Domain Alignment for Unsupervised Domain Adaptation`, ICML
  2020. https://proceedings.mlr.press/v119/jiang20d.html
- Lingjing Kong, Shaoan Xie, Weiran Yao, Yujia Zheng, Guangyi Chen, Petar
  Stojanov, Victor Akinwande, and Kun Zhang. `Partial disentanglement for
  domain adaptation`, ICML 2022.
  https://proceedings.mlr.press/v162/kong22a.html
- Wanxing Chang, Ye Shi, Hoang Duong Tuan, and Jingya Wang. `Unified Optimal
  Transport Framework for Universal Domain Adaptation`, NeurIPS 2022.
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/bda6843dbbca0b09b8769122e0928fad-Abstract-Conference.html
- Jian Shen, Yanru Qu, Weinan Zhang, and Yong Yu. `Wasserstein Distance Guided
  Representation Learning for Domain Adaptation`, AAAI 2018.
  https://ojs.aaai.org/index.php/AAAI/article/view/11784
- Yan Wang, Chuan-Xian Ren, Yi-Ming Zhai, You-Wei Luo, and Hong Yan.
  `Probability-Polarized Optimal Transport for Unsupervised Domain Adaptation`,
  AAAI 2024. https://ojs.aaai.org/index.php/AAAI/article/view/29493
