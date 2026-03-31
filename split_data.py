import os
import random
import shutil

# --- कॉन्फ़िगरेशन ---
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# DATA_DIR वह फ़ोल्डर है जिसके अंदर वर्तमान में क्लास फ़ोल्डर हैं
SOURCE_DIR = 'C:/Users/vaibh/Downloads/Kaggle/alzheimer_mri_preprocessed_dataset/raw' 
# यह वह नया फ़ोल्डर होगा जहाँ 'train' और 'val' फ़ोल्डर बनेंगे
TARGET_DIR = SOURCE_DIR 

SPLIT_RATIO = 0.8  # 80% ट्रेनिंग, 20% वेलिडेशन
CLASSES = ['Non_Demented', 'Moderate_Demented', 'Mild_Demented'] 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def split_data():
    """क्लास फ़ोल्डरों के अंदर इमेजेस को train और val फ़ोल्डरों में विभाजित करता है।"""
    print(f"Starting data split for classes: {CLASSES}")
    
    # सुनिश्चित करें कि target फ़ोल्डर मौजूद है
    if not os.path.exists(TARGET_DIR):
        print(f"Error: Source directory not found at {SOURCE_DIR}")
        return

    # ट्रेन और वैल फ़ोल्डर बनाएँ
    train_dir = os.path.join(TARGET_DIR, 'train')
    val_dir = os.path.join(TARGET_DIR, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    total_files = 0
    
    for class_name in CLASSES:
        # स्रोत (Source) और गंतव्य (Destination) पथ सेट करें
        source_class_path = os.path.join(SOURCE_DIR, class_name)
        
        if not os.path.exists(source_class_path):
            print(f"Warning: Class folder {source_class_path} not found. Skipping.")
            continue
            
        # इस क्लास के लिए train और val फ़ोल्डर बनाएँ
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # इमेजेस की सूची प्राप्त करें
        all_files = [f for f in os.listdir(source_class_path) if os.path.isfile(os.path.join(source_class_path, f))]
        random.shuffle(all_files) # इमेजेस को फेरबदल करें
        
        # विभाजन (split) बिंदु निर्धारित करें
        split_point = int(SPLIT_RATIO * len(all_files))
        
        train_files = all_files[:split_point]
        val_files = all_files[split_point:]
        
        print(f"Class {class_name}: Train ({len(train_files)} files), Val ({len(val_files)} files)")
        total_files += len(all_files)
        
        # इमेजेस को train फ़ोल्डर में कॉपी करें
        for f in train_files:
            src = os.path.join(source_class_path, f)
            dst = os.path.join(train_dir, class_name, f)
            shutil.copy(src, dst)
            
        # इमेजेस को val फ़ोल्डर में कॉपी करें
        for f in val_files:
            src = os.path.join(source_class_path, f)
            dst = os.path.join(val_dir, class_name, f)
            shutil.copy(src, dst)
            
    print("-" * 30)
    print(f"Data split successful! Total files processed: {total_files}")
    print(f"Check your {TARGET_DIR} folder for 'train' and 'val' subfolders.")
    print("-" * 30)


if __name__ == '__main__':
    split_data()
