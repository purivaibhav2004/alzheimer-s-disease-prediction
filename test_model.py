import torch
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --- 1. कॉन्फ़िगरेशन ---
# यह वही पथ है जिसका उपयोग आपने ट्रेनिंग के लिए किया था
DATA_DIR = 'C:/Users/vaibh/Downloads/Kaggle/alzheimer_mri_preprocessed_dataset/raw' 
NUM_CLASSES = 3
MODEL_LOAD_NAME = 'adni_3class_model.pth' # वह मॉडल जिसे हम लोड करेंगे
PHASE_TO_TEST = 'val' # हम वेलिडेशन डेटा पर टेस्ट कर रहे हैं

# क्लास नाम (वही जो फ़ोल्डर नाम थे)
CLASS_NAMES = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented'] 

# --- 2. डेटा ट्रांसफ़ॉर्मेशन (ट्रेनिंग जैसा) ---
data_transforms = {
    PHASE_TO_TEST: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 3. मॉडल लोड करना ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    """प्री-ट्रेंड EfficientNet मॉडल को लोड करता है और हमारे प्रशिक्षित वज़न डालता है।"""
    print("Loading model...")
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_ftrs = model._fc.in_features
    model._fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
    
    try:
        model.load_state_dict(torch.load(MODEL_LOAD_NAME, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"Model {MODEL_LOAD_NAME} loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Error: Model file {MODEL_LOAD_NAME} not found. Did you run train_adni_model.py?")
        return None
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return None

# --- 4. टेस्टिंग फ़ंक्शन ---
def test_model(model):
    if model is None:
        return
        
    print(f"\n--- Starting Evaluation on {PHASE_TO_TEST} set ---")

    # डेटा लोडिंग
    try:
        dataset = datasets.ImageFolder(os.path.join(DATA_DIR, PHASE_TO_TEST), data_transforms[PHASE_TO_TEST])
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # सुनिश्चित करें कि PyTorch द्वारा लोड किए गए क्लास नाम हमारे CLASS_NAMES से मेल खाते हैं
        # (ImageFolder इसे वर्णमाला क्रम में सॉर्ट करता है)
        
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return
        
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. मेट्रिक्स की गणना और प्रदर्शन
    print("\n--- Evaluation Metrics ---")
    
    # 5.1. Classification Report (Precision, Recall, F1-Score)
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    
    # 5.2. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    # Confusion Matrix का बेहतर विज़ुअलाइज़ेशन (यदि आप Jupyter/Colab में हैं)
    # यहाँ हम इसे सादे टेक्स्ट में प्रिंट कर रहे हैं
    print(cm)
    
    overall_accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    
    print("\nEvaluation complete.")


if __name__ == '__main__':
    # 1. सुनिश्चित करें कि आपने train_adni_model.py सफलतापूर्वक चलाया है और adni_3class_model.pth फ़ाइल मौजूद है।
    # 2. test_model.py चलाएँ।
    trained_model = load_model()
    if trained_model:
        test_model(trained_model)

    
