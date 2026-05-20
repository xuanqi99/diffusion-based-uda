# Diffusion-Based UDA

This repository is for research on using diffusion-based generative models to
improve unsupervised domain adaptation (UDA).

The current codebase includes runnable PyTorch ERM, DANN, AFN, CDAN, MDD, and
JAN baselines under `uda/`. These standard UDA methods are intended as clean
reference points for future diffusion-assisted experiments, such as
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

See `uda/README.md` for detailed dataset layout, list-file mode, and additional
run examples.
