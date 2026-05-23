# Diffusion-UDA Research Workspace

This directory is a research workspace for studying how diffusion-based
generative models can improve unsupervised domain adaptation (UDA). The current
implementation provides runnable PyTorch ERM, DANN, AFN, CDAN, MDD, JAN, GTA,
ADDA, and MCD baselines, which are useful as clean reference points before adding
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
`Deep Transfer Learning with Joint Adaptation Networks`. GTA trains with source
classification loss plus an auxiliary-classifier GAN branch whose generator and
discriminator provide adaptation gradients for source and unlabeled target
embeddings, following `Generate To Adapt`. ADDA first pretrains a source encoder
and classifier, then freezes them while adversarially training a separate target
encoder to fool a domain discriminator, following `Adversarial Discriminative
Domain Adaptation`. MCD trains a shared feature extractor with two classifier
heads, maximizes their target prediction discrepancy while preserving source
classification, then updates the feature extractor to minimize that discrepancy,
following `Maximum Classifier Discrepancy for Unsupervised Domain Adaptation`.

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
- `GTA`: generate-to-adapt training with an F-C classification stream and an
  auxiliary-classifier GAN stream over generated source-like images.
- `ADDA`: adversarial discriminative domain adaptation with separate source and
  target encoders trained in source-supervised and target-adversarial stages.
- `MCD`: maximum classifier discrepancy with two task classifiers that expose
  and then reduce target-domain prediction disagreement.

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

Outputs are written to `runs/` by default and include `config.json`,
`metrics.csv`, `checkpoint_last.pt`, and `best_target.pt` when target labels are
available.

## References

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
