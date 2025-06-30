# 🧠 Synthetic Data Generation for Robust Optical Character Verification in Industrial Laser Marking

![Generated Sample](documentation/repository_figures/fig_qualitative_results.png)

This repository contains the full implementation of the **Final Degree Project** titled **"Synthetic Data Generation for Robust Optical Character Verification in Industrial Laser Marking"**. The project introduces a modular, data-centric pipeline for generating realistic synthetic datasets using fine-tuned diffusion models and training an end-to-end YOLOv11-based detection system for traceability and quality assurance in industrial packaging environments.

## 📌 Project Overview

Industrial Optical Character Verification (OCV) systems often suffer from:

- ⚠️ Limited annotated and diverse datasets
- ⚠️ Imbalanced character usage
- ⚠️ Underrepresented error scenarios (e.g., laser beam defects)
- ⚠️ High cost of real data collection

To address these challenges, this project:

- ✅ Implements a synthetic data generation pipeline using **Stable Diffusion XL + ControlNet**
- ✅ Augments datasets with diverse characters, error cases, and backgrounds
- ✅ Trains a high-accuracy **YOLOv11** model for character recognition and code validation
- ✅ Simulates real-world printing defects for robust model generalization
- ✅ Evaluates quality through **CLIP similarity**, **quantitative metrics**, and a **user study**

---

## Repository Structure

```bash
Synthetic-Data-Generation-for-Robust-OCV/
├── data/ # Dataset samples (real and synthetic)
│ ├── real/ # Original industrial images
│ └── synthetic/ # Generated samples
├── models/ # Pretrained model weights
│ ├── controlnet/ # Fine-tuned ControlNet checkpoints
│ └── yolov11/ # Trained detection model
├── notebooks/ # Jupyter notebooks for exploration
├── scripts/ # Main processing scripts
│ ├── data_generation/ # Synthetic data pipeline
│ ├── training/ # Model training scripts
│ └── evaluation/ # Performance assessment
├── configs/ # Configuration files
├── docs/ # Project documentation
└── img/ # Figures and visualizations
```
## 👩‍💻 Author

- **Julia Garcia Torné**  
  Supervised by **David Castells Rufas**  
  Academic Year: 2024–2025  
  Universitat Autònoma de Barcelona

---

