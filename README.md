# Two Stage Leaf Health Classifier  
Raspberry Pi + Python + Machine Learning

This project implements a two stage computer vision pipeline that identifies whether an image contains a leaf, and if so, determines whether the leaf is healthy or diseased. Training is performed on a Mac, while real time testing and image capture are done on a Raspberry Pi using remote SSH.

The system uses HSV color histograms, green pixel segmentation, and edge detection to create feature vectors that are lightweight and efficient for Raspberry Pi hardware.

---

# Project Features

• Two-stage machine learning pipeline  
  1. Gate Model: Leaf vs. Not-Leaf  
  2. Health Model: Healthy vs. Diseased  

• Supports real time image capture using a Raspberry Pi camera  
• Automatically saves annotated images to the directory `~/leaf_captures/`  
• Command line interface for preparing datasets, training models, and running inference  

---

# Models Used

### 1. Gate Model (Leaf vs. Not-Leaf)
This project uses the one-class gate option.  
Model file: `leaf_gate_oneclass.pkl`  
This model is trained only on leaf images using a One-Class SVM.

### 2. Health Model (Healthy vs. Diseased)
This model uses a Random Forest classifier.  
Model file: `model.pkl`  
It is trained using healthy and diseased leaf images.

---

# Dataset

The project uses the PlantVillage dataset, which contains labeled plant leaf images across a wide range of species. A preparation script reorganizes the dataset into:

leaflab/

-- healthy/

-- diseased/

-- leaves_all/

The dataset can be downloaded here:  
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

# System Overview

### Stage A — Gate Model
Determines whether the image contains a leaf.  
The One-Class SVM produces either +1 (leaf) or -1 (not a leaf).

### Stage B — Health Model
Runs only if Stage A identifies the image as a leaf.  
Classifies the leaf as either Healthy (0) or Diseased (1).

### Feature Extraction
Each image is transformed into a feature vector with:
- HSV color histograms (50/60/60 bins)
- Green pixel ratio
- Edge ratio using Canny edge detection

This approach keeps computation requirements low and suitable for the Raspberry Pi CPU.

---

# Hardware Used

### Raspberry Pi  
- Raspberry Pi 5 model  
(Your configuration can be added if needed.)

### Camera  
- Raspberry Pi Camera Module v2.1  
  - 8 megapixel Sony IMX219 sensor  
  - Connected via CSI ribbon cable  
  - Captured using Picamera2 or OpenCV

---

# System Architecture
NEEDS TO ADDED

---

# Installation and Setup
1. Clone the Repository
2. Create and Activate Virtual Environment (Pi)
3. Install Dependencies
- Dependencies include:
  - numpy 1.24.4
  - scikit-learn 1.7.1
  - opencv-python 4.12.0.88
  - joblib

---

# Training Instructions - (Performed on Mac)
A. Prepare the Dataset:

python leaf_pipeline.py --prepare-plantvillage \
    --pv-root ~/Downloads/PlantVillage \
    --out-root ~/leaf_project/datasets/leaflab

B. Train the One Class Gate:

python leaf_pipeline.py --train-gate-oneclass \
    --gate-leaf-dir ~/leaf_project/datasets/leaflab/leaves_all \
    --gate-nu 0.05 --gate-gamma scale

C. Train the Health Classifier:

python leaf_pipeline.py --train-health \
    --healthy-dir ~/leaf_project/datasets/leaflab/healthy \
    --diseased-dir ~/leaf_project/datasets/leaflab/diseased

---

# Transferring Model and Testing on Raspberry Pi
- Transferring Model from the Mac terminal:
  - scp leaf_gate_oneclass.pkl <pi-username>@<pi-hostname>:~/leaf_project/
  - scp model.pkl <pi-username>@<pi-hostname>:~/leaf_project/

- Testing on Raspberry Pi:
  - Classify a Single Image:
    - python leaf_pipeline.py --image ~/leaf_project/myphoto.png
  - To run without opening a preview window (useful over SSH):
    - python leaf_pipeline.py --image ~/leaf_project/myphoto.png --no-show
  - Run One Capture from the Pi Camera:
    - python leaf_pipeline.py

- Output Files
  - Annotated images are automatically saved to: ~/leaf_captures/







