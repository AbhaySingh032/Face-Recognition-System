# 🧠 Face Recognition System using FaceNet, MTCNN & SVM

This project implements a face recognition system using state-of-the-art techniques in face detection and embedding. It demonstrates high-accuracy identity recognition on a small custom dataset of 5 celebrities.

# 📌 Key Components

Face Detection: MTCNN (Multi-task Cascaded Convolutional Networks)

Feature Extraction: FaceNet (pretrained model for 128D face embeddings)

Face Classification: Support Vector Machine (SVM)

# 🎯 Objective

To build an efficient, high-accuracy face recognition system capable of identifying known individuals using very limited training data by leveraging pretrained deep learning models.

# 📁 Dataset

Subjects: 5 celebrities

Samples per identity: ~10–15 images

Format: JPG/PNG face images (pre-cropped or full photos)

# ⚙️ Workflow

A[Input Image] --> B[MTCNN Face Detection]

B --> C[Face Alignment & Cropping]

C --> D[FaceNet Embedding Extraction]

D --> E[SVM Classifier]

E --> F[Predicted Identity]


# 🔍 Features

📷 Detects and crops faces from raw images using MTCNN

📐 Embeds faces into a 128D vector space using FaceNet

🎯 Classifies identities using a linear SVM with high accuracy

🔄 Easily extendable to more identities or real-time applications


# 📊 Results

Accuracy: 95%+ on test set with minimal samples

Inference Time: ~50–100ms per image (depending on GPU/CPU)

Model Generalization: Excellent performance even on unseen poses and lighting


# 🚀 Getting Started

**Clone the repository**

git clone https://github.com/AbhaySingh032/Face-Recognition-System.git

cd face-recognition-facenet-svm

**Install dependencies**
pip install -r requirements.txt

**Run the training script**
python train.py

**Run recognition on test images**

python predict.py --img_path path/to/image.jpg


# 📦 Requirements

Python 3.7+

TensorFlow or PyTorch (for FaceNet model)

MTCNN

Scikit-learn

NumPy, OpenCV, Matplotlib

# 📂 Project Structure

face-recognition/

├── data/

│   └── celebrity_faces/

├── models/

│   ├── facenet_model/

│   └── svm_classifier.pkl

├── train.py

├── predict.py

└── README.md


# 🛠️ Future Improvements

Add real-time webcam integration

Enable face clustering for unknown identities

Convert to web app (Flask/Streamlit)

Extend support to video-based recognition


# 👤 Author

Abhay Pal Singh

📧 rabhay032@gmail.com

# ⭐️ Show Some Love

If this project helped you or inspired you, give it a ⭐ and share it with others!

