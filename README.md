![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Status](https://img.shields.io/badge/Status-Completed-green)

# AgroXact: Lightweight Edge-AI Plant Disease Detection using Knowledge Distillation

AgroXact is a lightweight and deployable deep learning framework for plant disease detection and severity estimation, designed for real-time inference on edge devices such as smartphones, drones, and agribots.

---

## 🚀 Overview

* Developed a robust plant disease detection system for real-world agricultural environments
* Addressed domain gap by merging PlantVillage (lab) and PlantDoc (field) datasets
* Used Knowledge Distillation to compress EfficientNet-B4 into EfficientNet-B0
* Achieved high accuracy with significant model size reduction
* Enabled real-time inference and mobile deployment

---

## 🧠 Methodology

### 🔹 Dataset Construction

* Combined **PlantVillage + PlantDoc datasets**
* Created unified **34-class dataset**
* ~23,000+ images across diverse conditions 

### 🔹 Teacher Model

* EfficientNet-B4 (high-capacity model)
* Learned robust feature representations

### 🔹 Student Model

* EfficientNet-B0 (lightweight model)
* Optimized for edge deployment

### 🔹 Knowledge Distillation

* Evaluated multiple strategies:

  * Cross-Entropy
  * KL Divergence
  * Feature-based
  * Attention-based
  * Relational KD

* **Best Strategy:** Attention + Cross-Entropy

* Achieved **95.6% validation accuracy** 

---

## 📊 Results

* Student model surpassed teacher performance
* 3.5× reduction in parameters
* Real-time inference (~52 ms CPU latency) 
* Robust generalization across lab and field data

---

## 🌱 Disease Classes

Includes 34 plant disease categories such as:

* Tomato Septoria Leaf Spot
* Corn Rust Leaf
* Potato Early Blight
* Apple Scab
* Bell Pepper Bacterial Spot

(See `disease_labels.txt` for full list)

---

## 🌡 Severity Estimation

* Implemented confidence-based severity prediction
* Severity levels:

  * Healthy
  * Mild
  * Moderate
  * Severe 

---

## 📱 Deployment

* Model exported to **TorchScript**
* Integrated into Android application
* Supports:

  * Offline inference
  * Real-time detection
  * Field usage

---

## 💻 Code & Models

Due to size limitations:

🔗 Teacher Model [EfficientNet-B4](https://drive.google.com/file/d/1-hG50j0jmws4Ld1i8q8JoTQZDvxYN2lW/view?usp=sharing)
🔗 Student Model [EfficientNet-B0](https://drive.google.com/file/d/1txUNDlIJR_V6e0Rb6YpBBucv_o0bYD6Q/view?usp=sharing)
🔗 Mobile Model ([.pt](https://drive.google.com/file/d/1Fvn2Iy05YrVmXLpGjOongKMzYBSucqzF/view?usp=sharing))

---

## ⚙️ Tech Stack

* Python
* PyTorch
* EfficientNet
* OpenCV
* TorchScript
* Android Deployment

---

## 📄 Research Paper

📥 [View Paper](./Research_Paper.pdf)

---

## 🔬 Key Contributions

* Hybrid dataset for lab-to-field generalization
* Systematic comparison of 8 KD strategies
* Lightweight model for edge deployment
* Novel severity estimation without extra labels
* Real-time mobile deployment

---

## 🔗 Future Work

* Deploy on drones and IoT devices
* Pixel-level severity estimation
* Multilingual agricultural recommendations
* Integration with precision farming systems

---

## 💡 Key Highlights

* Research + Deployment + Mobile App
* Real-world agricultural impact
* Edge-AI optimized system
* Conference-level work

---
