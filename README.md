# Feedback Residual Dense Network (FRDN) for Image Super-Resolution

This repository contains the full implementation of a custom Feedback Residual Dense Network (FRDN), a novel architecture combining the strengths of Residual Dense Networks (RDN) and Feedback mechanisms inspired by SRFBN.

## 📌 Highlights

- 🧠 **Recurrent Feedback Loop** using shared RDN-based refinement blocks
- 🔁 Multi-step refinement with T-step unrolling
- 🧪 Training pipeline with L1 loss and PSNR/SSIM evaluation
- 🖼️ Visualizes input LR, predicted SR, and ground truth HR images

## 📂 Folder Structure

project/
├── frdn_arch.py # FRDN model
├── rdn_arch.py # Residual Dense Network base
├── srfbn_arch.py # Reference for feedback (not used directly)
├── train.py # Training loop
├── evaluate.py # Evaluation script (PSNR, SSIM)
├── utils.py # Utility functions (optional)
└── dataset/
├── train/
│ ├── LR/
│ └── HR/
└── test/
├── LR/
└── HR/


## More details about the project are in frdn_doc
