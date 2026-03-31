import torch
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
import os

# --- 1. कॉन्फ़िगरेशन ---
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# V.V.V.I: ADNI डेटासेट का वास्तविक पथ (यूजर द्वारा प्रदान किया गया)
# ImageFolder उम्मीद करता है कि इस फ़ोल्डर के अंदर 'train' और 'val' फ़ोल्डर हों।
DATA_DIR = 'C:/Users/vaibh/Downloads/Kaggle/alzheimer_mri_preprocessed_dataset/raw' 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

NUM_CLASSES = 3 # NC, MCI, AD
BATCH_SIZE = 32
NUM_EPOCHS = 10 
MODEL_SAVE_NAME = 'adni_3class_model.pth' # यह फ़ाइल Flask ऐप द्वारा लोड की जाएगी

# --- 2. डेटा ट्रांसफ़ॉर्मेशन ---
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 3. मॉडल को लोड करना और संशोधित करना ---
# GPU (CUDA) उपलब्ध होने पर उसका उपयोग करें, अन्यथा CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# EfficientNet B0 को PyTorch हब से लोड करें
try:
    model = EfficientNet.from_pretrained('efficientnet-b0')
except Exception as e:
    print(f"Error loading pre-trained EfficientNet B0. Check your internet connection or PyTorch installation. Error: {e}")
    exit()

# अंतिम वर्गीकरण परत (Fully Connected layer) को 3 आउटपुट क्लास के लिए संशोधित करें
num_ftrs = model._fc.in_features
model._fc = torch.nn.Linear(num_ftrs, NUM_CLASSES) 

model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 4. डेटा लोडिंग ---
# ImageFolder उम्मीद करता है कि DATA_DIR के अंदर 'train' और 'val' फ़ोल्डर हों, 
# और उन 'train'/'val' फ़ोल्डर के अंदर NC, MCI, AD जैसे क्लास फ़ोल्डर हों।
# उदाहरण संरचना: C:/.../raw/train/NC/img1.jpg
try:
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                      for x in ['train', 'val']} 
except Exception as e:
    print(f"Error loading datasets. Check if your DATA_DIR path is correct and contains 'train' and 'val' subfolders. Error: {e}")
    exit()

dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 यदि Windows पर एरर आ रही हो
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes # जैसे ['AD', 'MCI', 'NC']

print(f"\n--- Training Setup Complete ---")
print(f"Dataset found at: {DATA_DIR}")
print(f"Loaded dataset classes: {class_names}")
print(f"Train size: {dataset_sizes['train']}, Validation size: {dataset_sizes['val']}")
print("-" * 30)

# --- 5. ट्रेनिंग लूप ---
def train_model():
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch}/{NUM_EPOCHS - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # ट्रेनिंग मोड
            else:
                model.eval()   # मूल्यांकन मोड

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # डीप कॉपी मॉडल अगर यह सबसे अच्छा मॉडल है (val accuracy के आधार पर)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), MODEL_SAVE_NAME)
                print(f"--> Saved new best model with Acc: {best_acc:.4f}")

    print("\n" + "="*30)
    print(f'Training complete. Final best val Acc: {best_acc:.4f}')
    print(f'Model saved as {MODEL_SAVE_NAME} in the current directory.')
    print("="*30)


if __name__ == '__main__':
    train_model()
