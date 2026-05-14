# Diffusion-UDA Research Workspace

This directory is a research workspace for studying how diffusion-based
generative models can improve unsupervised domain adaptation (UDA). The current
implementation provides runnable PyTorch ERM and DANN baselines, which are
useful as clean reference points before adding diffusion-generated images,
diffusion-based feature regularization, or other generative adaptation
strategies.

ERM trains only on labeled source-domain images and evaluates on the target
domain when target labels are available. This makes it a practical baseline for
measuring whether future diffusion-assisted UDA components provide real gains.
DANN trains with source classification loss plus a domain-adversarial loss over
source and target images, while never using target class labels for training.

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

## Run Examples

OfficeHome:

```powershell
python uda/erm.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
```

OfficeHome with DANN:

```powershell
python uda/dann.py --data-root D:\datasets --dataset officehome --source Art --target Clipart --arch resnet50 --epochs 20 --batch-size 32
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

Outputs are written to `runs/` by default and include `config.json`,
`metrics.csv`, `checkpoint_last.pt`, and `best_target.pt` when target labels are
available.
