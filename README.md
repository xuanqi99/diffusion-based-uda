# Diffusion-Based UDA

This repository is for research on using diffusion-based generative models to
improve unsupervised domain adaptation (UDA).

The current codebase includes runnable PyTorch ERM, DANN, AFN, CDAN, MDD, JAN,
GTA, ADDA, MCD, and SymmNets baselines under `uda/`. These standard UDA methods are
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

See `uda/README.md` for detailed dataset layout, list-file mode, and additional
run examples.
