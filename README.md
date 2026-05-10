# Torch-Reimplementations-
This repository contains my implementations of various Deep Learning Models and Techniques in PyTorch. While the concepts and architectures are based on existing literature and online resources, all code has been written/re-written by me for learning and practice purposes.

# 🚀 Models & Implementations

---

## 1. Generative Adversarial Networks (DCGAN)

**File:** `DCGAN.ipynb`

### Model
Deep Convolutional GAN (DCGAN)

### Description
This notebook implements a DCGAN to generate realistic images. It utilizes strided convolutions in the Discriminator and fractional-strided convolutions in the Generator.

### Dataset
CelebA (Faces)

### Key Features
- Custom weight initialization
- Binary Cross Entropy (BCE) Loss tracking
- Visual progress monitoring of generated images through epochs

---

## 2. Object Detection & Segmentation

**File:** `detection.ipynb`

### Model
Mask R-CNN

### Description
Leveraging `torchvision`, this notebook implements a Mask R-CNN model with a ResNet-50 backbone for simultaneous object detection and instance segmentation.

### Dataset
Penn-Fudan Pedestrian Dataset (`PennFudanPed.zip`)

### Key Features
- Custom Dataset class for handling masks
- Bounding box visualization
- Fine-tuning a pre-trained COCO model for pedestrian detection tasks

---

## 3. Natural Language Processing (NLP)

**File:** `NLP.ipynb`

### Model
Sequence-to-Sequence (Seq2Seq) with Attention

### Description
A from-scratch implementation of an Encoder-Decoder architecture for character-level or word-level language tasks such as translation or name classification.

### Dataset
`Datasets/data/`
- `names`
- `eng-fra.txt`

### Key Features
- Gated Recurrent Units (GRU)
- Attention mechanisms
- Unicode-to-ASCII preprocessing

---

## 4. Hardware Optimization (CPU to GPU)

**File:** `cpu2gpu.ipynb`

### Tooling
PyTorch Profiler & Benchmark

### Description
A technical deep-dive into optimizing data transfer between Host (CPU) and Device (GPU).

### Key Features
- Comparison between Blocking vs. Non-Blocking copies
- Implementation of Pinned Memory for faster data throughput
- Multi-stream synchronization
- Profiling using `torch.profiler`

---

# 📊 Datasets Setup

To run these notebooks, ensure your `Datasets` folder is populated as follows:

```bash
Datasets/
│
├── celeba/
│   └── img_align_celeba/
│       └── [CelebA image files]
│
├── PennFudanPed.zip
│
└── data/
    ├── names/
    └── eng-fra.txt

## 📂 Repository Structure

```text
.
├── DCGAN.ipynb             # Generative Adversarial Network implementation
├── detection.ipynb         # Object Detection and Segmentation
├── NLP.ipynb               # Sequence-to-Sequence NLP models
├── cpu2gpu.ipynb           # Performance & Hardware Optimization
├── Datasets                # Local data storage (Ignored by Git LFS)
│   ├── PennFudanPed.zip    # Dataset for Detection
│   ├── celeba/             # Dataset for DCGAN
│   └── data/               # Text data for NLP tasks
└── README.md
