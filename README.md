# 🧠 Alzheimer’s Disease Detection using Deep Learning

## 📌 Project Overview
This project is a **Deep Learning-based web application** that predicts the stage of Alzheimer’s Disease using **MRI brain images**. The system allows users to upload MRI scans and get real-time predictions along with confidence scores.

---

## 🎯 Objective
The main objective of this project is to support **early detection of Alzheimer’s Disease**, which is difficult to diagnose in early stages due to complex patterns in MRI images.

---

## 🗂 Dataset
- Source: Kaggle (OASIS MRI Dataset)
- Total Images: 2,200+ MRI scans
- Categories:
  - Normal
  - Mild Cognitive Impairment (MCI)
  - Moderate Alzheimer’s
- Data Split:
  - 80% Training
  - 20% Testing

---

## ⚙️ Technologies Used
- **Python**
- **Flask** (Web Framework)
- **PyTorch**
- **EfficientNet-B0** (Transfer Learning)
- **PIL** (Image Processing)
- **SQLite** (User Authentication)
- **HTML, CSS, JavaScript** (Frontend)

---

## 🔄 Workflow
1. User uploads MRI image
2. Image is preprocessed:
   - Resizing (256 → 224)
   - Center Cropping
   - Normalization
3. Preprocessed image is passed to the trained model
4. Model predicts disease stage
5. Output is displayed with confidence score

---

## 🧠 Model Details
- Model: EfficientNet-B0
- Technique: Transfer Learning
- Input Size: 224x224
- Output Classes:
  - Normal
  - Mild Stage (MCI)
  - High Stage Alzheimer’s

---

## 📊 Performance Metrics
- Accuracy: **~94%**
- Precision: ~94%
- Recall: ~94%
- F1 Score: ~94%

---

## 🚀 Features
- User Login & Registration System
- MRI Image Upload
- Real-time Prediction
- Confidence Score Display
- Simple and User-Friendly Interface

## 📷 Screenshots

### 🔹 Home Page
![Home Page](<img width="1882" height="908" alt="Screenshot 2025-10-04 204108" src="https://github.com/user-attachments/assets/c0e9116d-163b-4f19-a06f-dca20c734b68" />

)

### 🔹 Prediction Result
![Prediction](<img width="1879" height="899" alt="Screenshot 2025-10-04 204247" src="https://github.com/user-attachments/assets/609f18f4-4c5c-4fd3-ad75-a6765cf3af8d" />

)

---

## 🧑‍💻 My Contribution
- Worked on **data preprocessing pipeline**
- Integrated trained deep learning model with Flask application
- Handled image input and prediction flow
- Implemented user authentication using SQLite
- Managed prediction output and response handling

---

## 📷 Sample Output
- Displays uploaded MRI image
- Shows predicted stage of Alzheimer’s
- Provides confidence percentage

---

## 🔮 Future Scope
- Use advanced models (EfficientNet B1–B7)
- Add multi-modal data (PET scans, clinical data)
- Deploy on cloud for real-world usage
- Improve accuracy with larger datasets

---

## 📌 Conclusion
This project demonstrates how deep learning can be used to **analyze medical image data** and assist in early diagnosis of Alzheimer’s Disease with high accuracy.

---

## 📎 How to Run
```bash
# Clone the repository
git clone https://github.com/purivaibhav2004/alzheimer-detection.git

# Navigate to project folder
cd alzheimer-detection

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
