# Diffusion-Based UDA

This repository is for research on using diffusion-based generative models to
improve unsupervised domain adaptation (UDA).

The current codebase includes runnable PyTorch ERM, DANN, AFN, CDAN, MDD, JAN,
CAN, GTA, ADDA, MCD, SymmNets, GVB-GD, ETD, SRDC, ACTIR, and TCM baselines under `uda/`. These standard UDA methods are
intended as clean reference points for future diffusion-assisted experiments,
such as
diffusion-generated target style images, synthetic source augmentation, or
generative feature regularization.

Supported dataset presets:

- `officehome`
- `office31`
- `visda2017`
- `domainnet`

## Quick Start

Install dependencies:

```powershell
pip install -r uda/requirements.txt
```

Run an OfficeHome ERM baseline:

```powershell
python uda/erm.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

Run an OfficeHome DANN baseline:

```powershell
python uda/dann.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

Run an OfficeHome AFN baseline:

```powershell
python uda/afn.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

Run an OfficeHome CDAN baseline:

```powershell
python uda/cdan.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

Run an OfficeHome MDD baseline:

```powershell
python uda/mdd.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

The MDD entry follows the margin disparity discrepancy baseline reported in
`A Closer Look at Smoothness in Domain Adversarial Training` (ICML 2022).

Run an OfficeHome JAN baseline:

```powershell
python uda/jan.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

The JAN entry follows `Deep Transfer Learning with Joint Adaptation Networks`
(ICML 2017) and aligns feature-prediction joint distributions with JMMD.

Run an OfficeHome CAN baseline:

```powershell
python uda/can.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

The CAN entry follows `Contrastive Adaptation Network for Unsupervised Domain
Adaptation` (CVPR 2019) with class-aware contrastive domain discrepancy over
source labels and target pseudo-labels.

Run an OfficeHome GTA baseline:

```powershell
python uda/gta.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

The GTA entry follows `Generate To Adapt: Aligning Domains using Generative
Adversarial Networks` (CVPR 2018) with a source classification stream and an
auxiliary-classifier GAN adaptation stream.

Run an OfficeHome ADDA baseline:

```powershell
python uda/adda.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --source-epochs 5 --epochs 20 --batch-size 32
```

The ADDA entry follows `Adversarial Discriminative Domain Adaptation`
(CVPR 2017) with separate source and target encoders and a domain
discriminator trained in two stages.

Run an OfficeHome MCD baseline:

```powershell
python uda/mcd.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

The MCD entry follows `Maximum Classifier Discrepancy for Unsupervised Domain
Adaptation` (CVPR 2018) with two classifier heads that maximize target
prediction discrepancy and a feature generator that minimizes it.

Run an OfficeHome SymmNets baseline:

```powershell
python uda/symmnets.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

The SymmNets entry follows `Domain-Symmetric Networks for Adversarial Domain
Adaptation` (CVPR 2019) with source/target task classifiers, a shared 2K
classifier, and two-level domain confusion.

Run an OfficeHome GVB-GD baseline:

```powershell
python uda/gvbgd.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

The GVB-GD entry follows `Gradually Vanishing Bridge for Adversarial Domain
Adaptation` (CVPR 2020) with generator-side and discriminator-side bridge
modules over an entropy-weighted domain-adversarial objective.

Run an OfficeHome ETD baseline:

```powershell
python uda/etd.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32 --optimizer adamw
```

The ETD entry follows `Enhanced Transport Distance for Unsupervised Domain
Adaptation` (CVPR 2020) with attention-reweighted transport costs,
Kantorovich potential networks, and target entropy minimization.

Run an OfficeHome SRDC baseline:

```powershell
python uda/srdc.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

The SRDC entry follows `Unsupervised Domain Adaptation via Structurally
Regularized Deep Clustering` (CVPR 2020) with target discriminative clustering,
feature-space clustering, source structural regularization, and optional soft
source sample weighting.

Run an OfficeHome ACTIR baseline:

```powershell
python uda/actir.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

The ACTIR-style entry follows `Invariant and Transportable Representations for
Anti-Causal Domain Shifts` with separate invariant and adaptive classifier
components, conditional decorrelation, an adaptive-gradient penalty, and
target-domain pseudo-label adaptation.

Run an OfficeHome TCM baseline:

```powershell
python uda/tcm.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

The TCM-style entry follows `Transporting Causal Mechanisms for Unsupervised
Domain Adaptation` (ICCV 2021) with learned mechanism proxies, interventional
prediction over mechanism priors, mechanism alignment, and target pseudo-label
adaptation.

See `uda/README.md` for detailed dataset layout, list-file mode, and additional
run examples.
