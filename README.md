# 🧠 Improving Evolutionary Neural Architecture Search for Image Classification Accuracy
---

## 📘 Overview

This project implements and extends **Evolutionary Neural Architecture Search (ENAS)** for **image classification**, focusing on improving accuracy and computational efficiency using **CIFAR-10** and **CIFAR-100** datasets.  

Traditional NAS methods are resource-intensive and slow. This work integrates **pretrained backbones**, **progressive evolution**, **Hyperband tuning**, and **search-space pruning** to design a faster and more accurate ENAS pipeline.

---

## 🚀 Key Contributions

- **Enhanced ENAS Framework:** Combines **aging evolution**, **weight sharing**, and **progressive search** to improve exploration and convergence.  
- **Pre-trained Knowledge Transfer:** Incorporates **MobileNetV2** and **custom deep CNNs** as frozen backbones to accelerate architecture discovery.  
- **Search-Space Reduction:** Filters low-performing architectures to focus computation on high-potential designs.  
- **Hyperparameter Optimization:** Utilizes **Keras Tuner (Hyperband / RandomSearch)** for efficient NAS exploration.  
- **Scalable Evaluation:** Provides modular testing across multiple CIFAR batches with performance visualization.

---

## 🧩 Project Structure

```bash
├── NASTest.py
│   ├─ RandomSearch NAS on CIFAR-10 (baseline NAS)
│   └─ Visualizes predicted vs actual classes.
│
├── ENASTest.py
│   ├─ NAS with pretrained MobileNetV2 backbone.
│   └─ Uses Hyperband for accelerated tuning.
│
├── ENAS_Hyperband_DeeperPretrain_Full.py
│   ├─ Loads full CIFAR-10 dataset.
│   ├─ Pre-trains deeper CNN for feature transfer.
│   ├─ Runs ENAS via Hyperband using pretrained backbone.
│   ├─ Retrains best model and evaluates on test data.
│   └─ Saves models and logs to `/nas_output/`.
