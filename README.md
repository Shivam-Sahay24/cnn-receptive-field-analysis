# cnn-receptive-field-analysis
# Visualizing the Effective Receptive Field (ERF) in CNNs 🧠

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## 📌 The Engineering Problem
In computer vision, a neural network must "see" a large enough portion of an image to understand context—this is the **Receptive Field**. Early deep learning architectures relied on massive, single-pass spatial filters (like 5x5 or 7x7) to achieve this. However, large filters are computationally expensive, making them difficult to deploy on constrained edge hardware (such as autonomous navigation or mobile vision systems).

This project mathematically and visually proves how modern CNN architectures manipulate tensor mathematics to maximize spatial awareness while minimizing parameter counts.

## 🔬 The 3-Types Of Architecutres in this experiment
Using **PyTorch** and a single image from the **CIFAR-10** dataset, this experiment isolates three distinct approaches to achieving a **7x7 Receptive Field** (reducing a 32x32 tensor to a 26x26 feature map).

1. **The Brute Force Era (Single 7x7):** The historical baseline.
2. **The Factorization Era (VGG-Style Stacked 3x3):** Stacking smaller layers to achieve the same boundary with deeper non-linearity.
3. **The Edge-Deployment Era (Dilated 3x3):** Using atrous convolutions to span a massive area with minimal weights.

---

## 📊 Visual Proof: Backpropagating the ERF
To prove how these networks "see" the data, I isolated the exact center pixel of the output feature map and ran backpropagation to calculate the gradients on the input image. 

**Here is the resulting 2D Gaussian Gradient Heatmap:**

<p align="center">
  <img src="https://raw.githubusercontent.com/Shivam-Sahay24/cnn-receptive-field-analysis/main/assets/output.jpg" alt="ERF Heatmap Analysis" width="100%">
</p>

### 🔑 Key Takeaways from the Visuals:
* **The Blunt 7x7 (Left):** Gives equal, rigid weight to every pixel in its bounding box.
* **The Stacked VGG (Center):** Demonstrates the emergence of a **Gaussian Distribution**. Deeper networks naturally learn to hyper-focus on the core of an object and smoothly taper off toward the periphery.
* **The Dilated 3x3 (Right):** Spans the entire 7x7 grid but mathematically skips the intermediate pixels—the ultimate cheat code for massive spatial reach.

## 🧮 The Parameter Math
While textbooks often state that stacked 3x3 filters are cheaper than large filters, analyzing the channel depth reveals the reality of dense networks:

| Architecture | Spatial Output | Total Learnable Parameters |
| :--- | :--- | :--- |
| **Single 7x7** | `[1, 64, 26, 26]` | 9,472 |
| **Stacked 3x3 (x3)** | `[1, 64, 26, 26]` | 75,648* |
| **Dilated 3x3 (d=3)** | `[1, 64, 26, 26]` | **1,792** |

*\*Note: The parameter explosion in the stacked VGG approach is due to the dense channel connections (64 -> 64) maintained across multiple layers, highlighting the engineering trade-off between deeper non-linear focus and memory cost.*

---

## 🚀 How to Run the Code
You can replicate this experiment and generate the heatmaps locally.

1. Clone the repository:
   ```bash
   git clone [https://github.com/Shivam-Sahay24/cnn-receptive-field-analysis.git](https://github.com/Shivam-Sahay24/cnn-receptive-field-analysis.git)
2. Install the requirements:
   ```bash
   pip install torch torchvision matplotlib seaborn

3. Run the analysis script:
   ```bash
   python erf_experiment.py
