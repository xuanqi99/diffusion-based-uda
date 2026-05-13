# Diffusion-Based UDA

This repository is for research on using diffusion-based generative models to
improve unsupervised domain adaptation (UDA).

The current codebase includes a runnable PyTorch ERM baseline under `uda/`.
ERM trains only on labeled source-domain images and evaluates on the target
domain when target labels are available. It is intended as a clean baseline for
future diffusion-assisted UDA experiments, such as diffusion-generated target
style images, synthetic source augmentation, or generative feature
regularization.

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

Run a smoke test without real data:

```powershell
python uda/erm.py --use-fake-data --dataset officehome --source Art --target Clipart --arch small_cnn --epochs 1 --steps-per-epoch 2 --batch-size 8 --eval-batch-size 8 --num-workers 0
```

Run an OfficeHome ERM baseline:

```powershell
python uda/erm.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

See `uda/README.md` for detailed dataset layout, list-file mode, and additional
run examples.
