import torch
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, matthews_corrcoef
import numpy as np
import os
# Fix: Import EfficientNet from the library used in your training scripts
from efficientnet_pytorch import EfficientNet 

# --- Configuration ---
# 1. YAHAN APNE TRAINING DATASET FOLDER KA PATH DENA HOGA (MANDATORY)
# सुनिश्चित करें कि यह पाथ आपके 'train' सबफ़ोल्डर तक जाता है
# EXAMPLE: 'C:/Users/vaibh/Downloads/Kaggle/alzheimer_mri_preprocessed_dataset/raw/train'
TRAIN_DATA_DIR = 'C:/Users/vaibh/Downloads/Kaggle/alzheimer_mri_preprocessed_dataset/raw/train' 

MODEL_PATH = 'adni_3class_model.pth' 
CLASS_NAMES = ["Mild_Demented", "Moderate_Demented", "Non_Demented"]
BATCH_SIZE = 32

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model structure definition (Same as used in training)
def define_model():
    """Defines and loads the EfficientNet model structure."""
    try:
        # FIX: Use EfficientNet.from_name from the efficientnet_pytorch library.
        # This method only builds the model structure ('efficientnet-b0') 
        # without attempting to download pre-trained weights, solving the hash error.
        model = EfficientNet.from_name('efficientnet-b0')
        
        # Modify the final classification layer for 3 classes
        # Note: 'efficientnet_pytorch' library uses '_fc' attribute
        num_ftrs = model._fc.in_features 
        model._fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
        
        # Load the saved fine-tuned weights
        print(f"Loading weights from {MODEL_PATH}...")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval() # Set model to evaluation mode
        return model
        
    except Exception as e:
        print(f"Error defining or loading model: {e}")
        # If the local model file loading fails, this will catch it too.
        return None

# Data transformations (Must match training preprocessing)
def get_data_transforms():
    """Returns the transformation pipeline for the input image."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def calculate_specificity_and_mcc(cm, all_targets, all_preds):
    """Calculates specificity for each class and overall MCC."""
    num_classes = cm.shape[0]
    specificities = []
    
    # Specificity Calculation: Specificity_i = TN_i / (TN_i + FP_i)
    # TN_i = Sum of all elements EXCEPT row i and column i
    # FP_i = Sum of column i EXCEPT element at (i, i)
    for i in range(num_classes):
        # Delete row i and column i to get the TN block for class i
        TN = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        # FP is the sum of the predicted values for class i (column i) minus the diagonal (TP)
        FP = np.sum(cm[:, i]) - cm[i, i]
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        specificities.append(specificity)
        
    # MCC Calculation (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(all_targets, all_preds)

    return specificities, mcc

# Main evaluation function
def evaluate_training_set():
    """Runs the model on the training dataset and calculates all metrics."""
    # Check if the training data path exists
    if not os.path.exists(TRAIN_DATA_DIR):
        print(f"\nERROR: Training data directory not found at: {TRAIN_DATA_DIR}")
        print("Please update the TRAIN_DATA_DIR variable with the correct path.")
        return

    # 1. Load Model
    model = define_model()
    if model is None:
        return

    # 2. Load Data (ImageFolder expects class subfolders inside TRAIN_DATA_DIR)
    data_transforms = get_data_transforms()
    train_dataset = datasets.ImageFolder(TRAIN_DATA_DIR, data_transforms)
    
    if len(train_dataset) == 0:
        print(f"\nERROR: Found 0 images in {TRAIN_DATA_DIR}. Check the folder structure.")
        return
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    all_preds = []
    all_targets = []
    
    print("\n--- Starting Training Set Evaluation (This may take longer than testing) ---")
    
    # 3. Predict on Training Set
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 4. Calculate Metrics
    overall_accuracy = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    
    # Specificity and MCC calculation
    specificities, mcc = calculate_specificity_and_mcc(cm, all_targets, all_preds)
    
    # Classification Report (Precision, Recall/Sensitivity, F1-Score)
    cr = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4, output_dict=True)
    
    
    # --- Print Results ---
    print("\n==================================================================")
    print("                 FINAL EVALUATION RESULTS (TRAINING SET)")
    print("==================================================================")
    print(f"Total Training Samples: {len(all_targets)}")
    
    # Extract weighted averages for overall paper table
    weighted_precision = cr['weighted avg']['precision']
    weighted_recall = cr['weighted avg']['recall']
    weighted_f1 = cr['weighted avg']['f1-score']
    
    # Calculate average specificity for the table (average of all 3 specificities)
    average_specificity = np.mean(specificities)


    print("\n1. Key Weighted Metrics for Comparative Table:")
    print("---------------------------------------------")
    print(f"Weighted Precision: {weighted_precision:.4f} ({(weighted_precision*100):.2f}%)")
    print(f"Weighted Sensitivity (Recall): {weighted_recall:.4f} ({(weighted_recall*100):.2f}%)")
    print(f"Weighted F1-Score: {weighted_f1:.4f} ({(weighted_f1*100):.2f}%)")
    print(f"Average Specificity: {average_specificity:.4f} ({(average_specificity*100):.2f}%)")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    
    print("\n2. Full Classification Report (Precision, Sensitivity/Recall, F1-Score):")
    print("----------------------------------------------------------------------")
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4))


if __name__ == '__main__':
    evaluate_training_set()
