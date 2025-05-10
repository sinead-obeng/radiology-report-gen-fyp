import os
import re
import sys

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import BioGptTokenizer

# Add project root to system path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
)

from config import (
    MEDICAL_ABBREVIATIONS,
    DATASET_DIR,
    STUDY_FILE,
    RECORD_FILE,
    OUTPUT_DIR,
)

# Download necessary NLTK components
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Load BioGPT tokenizer
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

# Define stop words, preserving medically relevant ones
STOP_WORDS = set(stopwords.words('english')) - {
    "I", "we", "you", "he", "she", "it", "they", "am", "is", "are", "was", "were",
    "be", "been", "have", "has", "had", "do", "does", "did", "a", "an", "the",
    "and", "but", "if", "or", "because", "as", "for", "with", "about", "between",
    "into", "through", "during", "before", "after", "on", "in", "out", "under",
    "over", "again", "further", "then", "once", "too", "very", "just", "can",
    "will", "should", "might", "must", "normal", "abnormal", "mild", "severe",
    "not", "without"
}

# Initialise lemmatizer
lemmatizer = WordNetLemmatizer()


# Text Cleaning and Tokenization Functions
def clean_text(report_text):
    """
    Cleans the medical report by:
    - Removing unnecessary sections (history, comparison)
    - Standardizing abbreviations
    - Preserving meaningful numbers
    - Tokenizing & lemmatizing words
    - Removing stop words while keeping essential ones

    Args:
        report_text (str): Raw medical report text.

    Returns:
        str: Processed and cleaned report text.
    """
    # Remove history and comparison sections
    report_text = re.sub(r'(?i)history:.*?(?=\n[A-Z ]+:|$)', '', report_text, flags=re.DOTALL)
    report_text = re.sub(r'(?i)comparison:.*?(?=\n[A-Z ]+:|$)', '', report_text, flags=re.DOTALL)

    # Remove special characters but keep medical symbols
    report_text = re.sub(r'[^a-zA-Z0-9.,/%\-? ]', '', report_text)

    # Convert to lowercase
    report_text = report_text.lower()

    # Standardise medical abbreviations
    for abbr, full_form in MEDICAL_ABBREVIATIONS.items():
        report_text = re.sub(rf'\b{re.escape(abbr)}\b', full_form, report_text)

    # Replace '//' with 'and'
    report_text = report_text.replace("//", " and ")

    # Remove age mentions
    report_text = re.sub(r'\b\d{1,2}\s?(year|years)\s?(old|ago)?\b', '', report_text)

    # Tokenize, lemmatize, and remove irrelevant stop words
    words = word_tokenize(report_text)
    cleaned_words = [
        lemmatizer.lemmatize(word) for word in words
        if word not in STOP_WORDS or word.isdigit()
    ]

    return " ".join(cleaned_words)


def tokenize_report(report_text):
    """
    Tokenizes medical reports using BioGPT tokenizer.

    Args:
        report_text (str): Cleaned medical report text.

    Returns:
        dict: Tokenized report containing input_ids and attention_mask.
    """
    tokens = tokenizer(
        report_text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    if tokens['input_ids'].shape[1] != 512:
        print(f"Warning: Report is of length {tokens['input_ids'].shape[1]}, truncated to 512.")

    return tokens


# Main Dataset Management Functions
def clean_medical_report(dataset_dir):
    """
    Cleans and tokenizes all medical reports in the dataset directory.

    Args:
        dataset_dir (str): Path to dataset directory.
    """
    for subset in ["train", "validation", "test"]:
        subset_dir = os.path.join(dataset_dir, subset)
        for root, _, files in os.walk(subset_dir):
            for file in files:
                if file.endswith(".txt"):
                    report_path = os.path.join(root, file)

                    with open(report_path, "r", encoding="utf-8") as f:
                        raw_text = f.read()

                    cleaned_text = clean_text(raw_text)
                    tokens = tokenize_report(cleaned_text)

                    tokenized_path = report_path.replace(".txt", "_tokens.npy")
                    np.save(tokenized_path, tokens["input_ids"].numpy())

                    print(f"Tokenized report saved: {tokenized_path}")


def create_split_mappings(study_file, record_file, dataset_dir, output_dir):
    """
    Creates mapping CSVs for train, validation, and test splits.

    Args:
        study_file (str): Path to study metadata file.
        record_file (str): Path to record metadata file.
        dataset_dir (str): Path to dataset directory.
        output_dir (str): Directory to save mapping CSVs.
    """
    # Load study and record data
    study_df = pd.read_csv(study_file).rename(columns={'path': 'report_path'})
    record_df = pd.read_csv(record_file).rename(columns={'path': 'image_path'})

    # Merge datasets
    merged_df = pd.merge(study_df, record_df, on=["subject_id", "study_id"])

    # Extract relative image path
    merged_df['relative_image_path'] = merged_df['image_path'].str.split('files/p1[01]/', n=1).str[1]

    # Generate tokenized report paths
    merged_df['tokenized_report_paths'] = merged_df.apply(
        lambda row: f"p{row['subject_id']}/s{row['study_id']}_tokens.npy", axis=1
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create mappings for each split
    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        split_output_file = os.path.join(output_dir, f"{split}_mapping.csv")

        split_image_paths = []
        for root, _, files in os.walk(split_dir):
            for file in files:
                if file.endswith('.dcm'):
                    file_path = os.path.relpath(os.path.join(root, file), dataset_dir).replace('\\', '/')
                    split_image_paths.append(file_path.split('/', 1)[1])

        split_df = merged_df[merged_df['relative_image_path'].isin(split_image_paths)]

        split_df.to_csv(split_output_file, index=False)
        print(f"{split} split mapping saved: {split_output_file}")


# Main Entry Point
def main():
    """
    Main function to clean reports and create split mappings.
    """
    # Step 1: Clean and tokenize all medical reports
    clean_medical_report(DATASET_DIR)

    # Step 2: Create dataset split mappings
    create_split_mappings(
        STUDY_FILE, RECORD_FILE,
        DATASET_DIR, os.path.join(OUTPUT_DIR, "mappings")
    )


if __name__ == "__main__":
    main()
