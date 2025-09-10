# Non-Target Boundary Attack GUI Overview

This project provides an interactive web application to visualize the **Non-Target Boundary Attack**, a type of adversarial attack against deep learning image classifiers.

The application allows a user to upload an image and observe in real-time as it is minimally perturbed, step-by-step, to cross the model's decision boundary and cause a misclassification. This implementation uses a pre-trained **ResNet50** model and is based on the logic from https://github.com/hang07020/non_target_boundary_attack

---

## Features
- **Interactive Interface**: Easily upload an image and configure attack parameters through a simple web UI.  
- **Real-Time Visualization**: Watch the adversarial sample evolve every 10 steps of the attack.  
- **Customizable**: Set the total number of steps to control the attack's duration and precision.  

---

## How to Run

### Prerequisites
- Python 3.9+  
- `pip` and `venv`  

### Setup and Installation
Clone the Repository:
```bash
git clone [your-repo-url]
cd non_target_boundary_attack_GUI
```

Create and Activate Virtual Environment:
```bash
# Create the environment
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Install Dependencies:
```bash
pip install streamlit numpy Pillow tensorflow keras opencv-python
```

### Launch the Application
Run the Streamlit app from your terminal:
```bash
streamlit run app.py
```

A new tab should open in your default web browser with the application running.

---

## Technology Stack
- **Backend**: Python  
- **GUI Framework**: Streamlit  
- **Deep Learning**: TensorFlow / Keras (with a pre-trained ResNet50 model)  
- **Image Processing**: Pillow, OpenCV  
