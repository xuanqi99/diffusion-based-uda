# Diffusion-UDA Research Workspace

This directory is a research workspace for studying how diffusion-based
generative models can improve unsupervised domain adaptation (UDA). The current
implementation provides runnable PyTorch ERM, DANN, AFN, CDAN, MDD, and JAN
baselines, which are useful as clean reference points before adding
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
`Deep Transfer Learning with Joint Adaptation Networks`.

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

Outputs are written to `runs/` by default and include `config.json`,
`metrics.csv`, `checkpoint_last.pt`, and `best_target.pt` when target labels are
available.

## References

- Mingsheng Long, Han Zhu, Jianmin Wang, and Michael I. Jordan. `Deep Transfer
  Learning with Joint Adaptation Networks`, ICML 2017.
  https://proceedings.mlr.press/v70/long17a.html
- Harsh Rangwani, Sumukh K Aithal, Mayank Mishra, Arihant Jain, and R.
  Venkatesh Babu. `A Closer Look at Smoothness in Domain Adversarial Training`,
  ICML 2022. https://proceedings.mlr.press/v162/rangwani22a.html
