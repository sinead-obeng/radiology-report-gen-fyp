import gc
import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np
import pydicom
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import BioGptForCausalLM, BioGptTokenizer
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATASET_DIR,
    MODELS_DIR,
    FEATURES_TRAIN_DIR,
    REPORTS_TRAIN_DIR,
    FEATURES_VALIDATION_DIR,
    REPORTS_VALIDATION_DIR,
)

# Configuration & Device Setup
BATCH_SIZE = 2
device = torch.device("cpu")

# Load Pretrained GPT Model & Tokenizer
gpt_model = BioGptForCausalLM.from_pretrained("microsoft/biogpt").to(device)
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

# Resize GPT token embeddings to handle new tokens if needed
gpt_model.resize_token_embeddings(len(tokenizer))

# ResNet-34 Feature Extractor Setup
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device)
return_nodes = {"avgpool": "features"}
feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

# Image transformation (ResNet-34 expects 3-channel input)
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Resize((224, 224)),  # Ensure correct input size for ResNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])


def extract_features_from_dicom(dicom_path):
    """
    Extract CNN features from a DICOM image.

    Args:
        dicom_path (str): Path to the DICOM (.dcm) file.

    Returns:
        np.ndarray: Extracted feature vector.
    """
    dicom_data = pydicom.dcmread(dicom_path)
    image_array = dicom_data.pixel_array.astype(np.float32)

    image_tensor = torch.tensor(image_array).unsqueeze(0).expand(3, -1, -1)
    image_tensor = transform(image_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(image_tensor)["features"]

    gc.collect()

    return features.squeeze().cpu().numpy()


def process_dataset(dataset_dir):
    """
    Process the dataset by extracting features from DICOM files and saving them as .npy files.

    Args:
        dataset_dir (str): Path to the main dataset directory.

    Returns:
        None
    """
    for subset in ["train", "validation", "test"]:
        subset_dir = os.path.join(dataset_dir, subset)
        feature_output_dir = os.path.join(dataset_dir, "features", subset)
        os.makedirs(feature_output_dir, exist_ok=True)

        for root, _, files in os.walk(subset_dir):
            for file in files:
                if file.endswith(".dcm"):
                    dicom_path = os.path.join(root, file)

                    patient_folder = os.path.basename(os.path.dirname(root))
                    patient_scan_folder = os.path.basename(root)

                    patient_feature_dir = os.path.join(feature_output_dir, patient_folder, patient_scan_folder)
                    os.makedirs(patient_feature_dir, exist_ok=True)

                    feature_path = os.path.join(patient_feature_dir, file.replace(".dcm", ".npy"))

                    if os.path.exists(feature_path):
                        continue

                    features = extract_features_from_dicom(dicom_path)
                    np.save(feature_path, features)

                    print(f"Saved features for {dicom_path} to {feature_path}")


class MedicalDataset(Dataset):
    """
    Custom Dataset class for loading medical features and corresponding tokenized reports.
    """

    def __init__(self, feature_dir, report_dir):
        """
        Args:
            feature_dir (str): Directory containing extracted feature files (.npy).
            report_dir (str): Directory containing tokenized report files (.npy).
        """
        self.file_paths = []
        for patient_folder in os.listdir(feature_dir):
            patient_folder_path = os.path.join(feature_dir, patient_folder)
            if os.path.isdir(patient_folder_path):
                for scan_folder in os.listdir(patient_folder_path):
                    scan_folder_path = os.path.join(patient_folder_path, scan_folder)
                    if os.path.isdir(scan_folder_path):
                        for file in os.listdir(scan_folder_path):
                            if file.endswith(".npy"):
                                feature_path = os.path.join(scan_folder_path, file)
                                report_path = os.path.join(report_dir, patient_folder, scan_folder + "_tokens.npy")
                                if os.path.exists(report_path):
                                    self.file_paths.append((feature_path, report_path))

        if not self.file_paths:
            raise RuntimeError("No valid samples found! Check that both features and reports exist.")

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (cnn_features, tokenized_report)
        """
        feature_path, report_path = self.file_paths[idx]
        cnn_features = np.load(feature_path)
        tokenized_report = np.load(report_path).squeeze()

        cnn_features = torch.tensor(cnn_features, dtype=torch.float32)
        tokenized_report = torch.tensor(tokenized_report, dtype=torch.long)

        return cnn_features, tokenized_report


def get_data_loaders(batch_size):
    """
    Get DataLoader objects for training and validation datasets.

    Args:
        batch_size (int): Number of samples per batch.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    train_dataset = MedicalDataset(FEATURES_TRAIN_DIR, REPORTS_TRAIN_DIR)
    val_dataset = MedicalDataset(FEATURES_VALIDATION_DIR, REPORTS_VALIDATION_DIR)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


def validate_model(model, val_loader, device):
    """
    Validate the model and compute the average validation loss.

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for validation dataset.
        device (torch.device): Device to run the model on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for cnn_features, tokenized_reports in val_loader:
            cnn_features = cnn_features.to(device)
            tokenized_reports = tokenized_reports.to(device)

            outputs = model(tokenized_reports, labels=tokenized_reports)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


def train_model():
    """
    Train the BioGPT model with CNN features, including validation and early stopping.

    Args:
        None

    Returns:
        None
    """
    train_loader, val_loader = get_data_loaders(BATCH_SIZE)

    optimizer = optim.AdamW(gpt_model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    num_epochs = 10
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        gpt_model.train()
        running_loss = 0.0

        print(f"Starting Epoch {epoch + 1}/{num_epochs}...")

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (cnn_features, tokenized_reports) in progress_bar:
            cnn_features = cnn_features.to(device)
            tokenized_reports = tokenized_reports.to(device)

            optimizer.zero_grad()
            outputs = gpt_model(tokenized_reports, labels=tokenized_reports)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(gpt_model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        val_loss = validate_model(gpt_model, val_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(MODELS_DIR, f"biogpt_medical_{timestamp}.pth")
            torch.save(gpt_model.state_dict(), save_path)
            print(f"Model improved. Saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete!")


if __name__ == "__main__":
    # Uncomment if feature extraction is needed before training
    print("Extracting features from dataset...")
    process_dataset(DATASET_DIR)
    print("Feature extraction complete!")

    print("Training model...")
    train_model()
    print("Training complete!")
