# Diffusion-Based UDA

This repository is for research on using diffusion-based generative models to
improve unsupervised domain adaptation (UDA).

The current codebase includes runnable PyTorch ERM, DANN, AFN, and CDAN
baselines under `uda/`. These standard UDA methods are intended as clean
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

See `uda/README.md` for detailed dataset layout, list-file mode, and additional
run examples.
