import os
import re
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import BioGptForCausalLM, BioGptTokenizer
import pydicom
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_DIR, REPORTS_DIR

# Path to saved BioGPT model
MODEL_PATH = "outputs/models/biogpt_medical_20250310-131418.pth"
device = torch.device("cpu")


def load_biogpt_model():
    """
    Load the BioGPT model and tokenizer.

    Returns:
        tuple: A tuple containing the loaded BioGPT model and tokenizer.
    """
    biogpt_model = BioGptForCausalLM.from_pretrained("microsoft/biogpt").to(device)
    biogpt_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    biogpt_model.eval()

    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    tokenizer.pad_token = tokenizer.eos_token

    print("BioGPT Model loaded successfully!")
    return biogpt_model, tokenizer


def load_resnet34():
    """
    Load the ResNet34 CNN model for feature extraction.

    Returns:
        nn.Module: The ResNet34 model configured for feature extraction.
    """
    resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device)
    return_nodes = {"avgpool": "features"} 
    cnn_model = create_feature_extractor(resnet, return_nodes=return_nodes)
    cnn_model.eval()
    return cnn_model


def transform_image(dicom_file):
    """
    Preprocess and transform a DICOM image for CNN input.

    Args:
        dicom_file (str): Path to the DICOM (.dcm) file.

    Returns:
        tuple: A tuple containing the DICOM dataset and transformed image tensor.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dicom_data = pydicom.dcmread(dicom_file)
    image_array = dicom_data.pixel_array.astype(np.float32)

    # Convert NumPy array to PIL Image
    image_pil = Image.fromarray(image_array).convert("L")
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    return dicom_data, image_tensor


def extract_cnn_features(cnn_model, image_tensors):
    """
    Extract CNN features from one or two DICOM images.

    Args:
        cnn_model (nn.Module): Pretrained CNN feature extractor.
        image_tensors (list): List of one or two transformed image tensors.

    Returns:
        torch.Tensor: Extracted feature tensor.
    """
    features_list = []
    with torch.no_grad():
        for image_tensor in image_tensors:
            cnn_output = cnn_model(image_tensor)
            cnn_features = cnn_output['features'].squeeze(0)
            features_list.append(cnn_features.to(torch.float32))

    if len(features_list) == 2:
        # Combine features from two images (e.g., average)
        combined_features = (features_list[0] + features_list[1]) / 2
    else:
        combined_features = features_list[0]

    return combined_features


def generate_report(biogpt_model, tokenizer, dicom_data, cnn_features):
    """
    Generate a structured radiology report (FINDINGS and IMPRESSION).

    Args:
        biogpt_model (BioGptForCausalLM): Loaded BioGPT model.
        tokenizer (BioGptTokenizer): Corresponding BioGPT tokenizer.
        dicom_data (pydicom.dataset.FileDataset): Metadata from DICOM file.
        cnn_features (torch.Tensor): Extracted features from CNN.

    Returns:
        str: Final structured radiology report.
    """
    # Extract procedure description or fallback
    procedure_description = dicom_data.get(
        (0x0040, 0x0254), 'Unknown procedure'
    ).value if dicom_data.get((0x0040, 0x0254)) else "Unknown procedure"

    # Generate the FINDINGS section
    findings_prompt = f"""
    FINAL REPORT
    EXAMINATION: {procedure_description}
    FINDINGS:
    """.strip()

    findings_input_ids = tokenizer.encode(findings_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        findings_output = biogpt_model.generate(
            input_ids=findings_input_ids,
            max_length=150,
            num_return_sequences=1,
            do_sample=True,
            top_k=20,
            top_p=0.95,
            temperature=0.3
        )

    findings_text = tokenizer.decode(
        findings_output[0], skip_special_tokens=True
    ).replace("<pad>", "").strip()

    findings_text = re.sub(
        r"FINAL REPORT\s+EXAMINATION:.*?FINDINGS:", "", findings_text, flags=re.IGNORECASE
    ).strip()

    findings_content = findings_text.split("IMPRESSION:")[0].strip()

    # Generate the IMPRESSION section
    impression_prompt = f"""
    {findings_content}
    IMPRESSION:
    """.strip()

    impression_input_ids = tokenizer.encode(impression_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        impression_output = biogpt_model.generate(
            input_ids=impression_input_ids,
            max_length=300,
            num_return_sequences=1,
            do_sample=True,
            top_k=20,
            top_p=0.95,
            temperature=0.3
        )

    impression_text = tokenizer.decode(
        impression_output[0], skip_special_tokens=True
    ).replace("<pad>", "").strip()

    impression_content = impression_text.replace(findings_content, "").strip()

    # Final report combining both sections
    final_report = f"""
    FINAL REPORT
    EXAMINATION: {procedure_description}

    FINDINGS: {findings_content}

    {impression_content}
    """.strip()

    return final_report


def process_study_folder(study_folder, biogpt_model, tokenizer, cnn_model):
    """
    Process a study folder with one or two DICOM files for report generation.

    Args:
        study_folder (str): Path to the study folder containing DICOM files.
        biogpt_model (BioGptForCausalLM): Loaded BioGPT model.
        tokenizer (BioGptTokenizer): Corresponding BioGPT tokenizer.
        cnn_model (nn.Module): CNN feature extractor model.

    Returns:
        None
    """
    dicom_files = [
        os.path.join(study_folder, f) for f in os.listdir(study_folder) if f.endswith('.dcm')
    ]

    if not dicom_files:
        print(f"No DICOM files found in {study_folder}")
        return

    print(f"Processing {len(dicom_files)} DICOM file(s) in {study_folder}...")

    dicom_data_list = []
    image_tensors = []

    # Process up to 2 DICOM images
    for dicom_file in dicom_files[:2]:
        dicom_data, image_tensor = transform_image(dicom_file)
        dicom_data_list.append(dicom_data)
        image_tensors.append(image_tensor)

    cnn_features = extract_cnn_features(cnn_model, image_tensors)

    # Generate report based on the first DICOM metadata
    report = generate_report(biogpt_model, tokenizer, dicom_data_list[0], cnn_features)

    # Extract IDs
    patient_id = dicom_data_list[0].get(
        (0x0010, 0x0020), 'Unknown Patient ID'
    ).value if dicom_data_list[0].get((0x0010, 0x0020)) else "Unknown_Patient"

    study_id = os.path.basename(study_folder)

    # Save the report
    output_dir = os.path.join(REPORTS_DIR, "test", f"p{patient_id}")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{study_id}.txt")
    with open(output_file, 'w') as file:
        file.write(report)

    print(f"Report saved: {output_file}")


def process_dataset(dataset_path):
    """
    Walk through a dataset directory and process each study folder.

    Args:
        dataset_path (str): Root path to the dataset directory.

    Returns:
        None
    """
    biogpt_model, tokenizer = load_biogpt_model()
    cnn_model = load_resnet34()

    for patient_folder in os.listdir(dataset_path):
        patient_path = os.path.join(dataset_path, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        for study_folder in os.listdir(patient_path):
            study_path = os.path.join(patient_path, study_folder)
            if os.path.isdir(study_path):
                process_study_folder(study_path, biogpt_model, tokenizer, cnn_model)


if __name__ == "__main__":
    process_dataset(TEST_DIR)
