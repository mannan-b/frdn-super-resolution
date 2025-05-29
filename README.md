
# FRDN: Feedback Residual Dense Network for Image Super-Resolution

A deep learning-based feedback super-resolution model that refines low-resolution images through a recurrent feedback loop using Residual Dense Blocks (RDBs), channel attention, and an iterative refinement process.

## 🚀 Highlights

- 🧠 **Feedback Loop**: Multi-iteration refinement of features.
- 🧩 **Residual Dense Blocks (RDBs)**: Deep local feature learning with dense and residual connections.
- 🎯 **Channel Attention**: Adaptively weights channels to focus on informative features.
- 🔁 **Shared Refinement Blocks**: Efficient parameter usage across iterations.
- 📈 **Per-iteration Supervision**: Intermediate outputs supervised for stable learning.
- 🧪 **Future Experimental Hooks**: Easily extensible for attention, quantization, pruning, or adversarial optimization.

## 📐 Architecture Overview

```
Input → SFENet (Conv x2) → Feedback Loop (T iterations)
      ├── [F₀, Hₜ₋₁] → 1x1 Conv → RDBs → CAB → Hₜ
      └── Hₜ → UPNet → Iₛᵣᵗ = Bicubic(LR) + Residual
```

Multiple iterations generate intermediate super-resolved outputs, improving quality progressively.

## 🛠️ Installation

```
git clone https://github.com/mannan-b/frdn-super-resolution.git
cd frdn-super-resolution
pip install -r requirements.txt
```

## 📂 Dataset Structure

Organize your data like this:

```
├── frdn_arch.py # FRDN model
├── rdn_arch.py # Residual Dense Network base
├── srfbn_arch.py # Reference for feedback (not used directly)
├── hat_arch.py # For CAB layer
data/
├── train/
│   ├── lr/  # Low-res training images
│   └── hr/  # High-res training images
└── test/
    ├── lr/
    └── hr/
```

## 📊 Sample Results

![sr_output](https://github.com/user-attachments/assets/4f4ecbf9-d6f7-4dbd-9bc3-8622e9a0b3ed)


## 🔬 Research Directions & Future Work

> Our goal is not just better resolution, but smarter reconstruction.

- 🧠 **Curriculum Learning** — Train on easy-to-hard examples based on upscaling complexity.
- 🎭 **Adversarial Feedback** — Treat each iteration as a generator, with a discriminator supervising realism.
- 🧩 **Masked Image Modeling** — Pretrain the model by reconstructing randomly masked regions for semantic awareness.

## 🧑‍💻 Author

Made with ❤️ by [@mannan-b](https://github.com/mannan-b)

## 🪄 Contributions Welcome!

Pull requests, issues, and forks are encouraged! Let’s build future-ready SR together.
