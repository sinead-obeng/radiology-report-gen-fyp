import os
import pydicom
import random
import shutil
import sys
import numpy as np

from scipy.ndimage import zoom, rotate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import P10_DIR, DATASET_DIR


# Helper Functions
def get_patient_ids(base_dir):
    """
    Retrieve patient IDs from the dataset directory structure.

    Args:
        base_dir (str): Path to the directory containing patient subfolders.

    Returns:
        list: List of patient IDs (folder names).
    """
    patient_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    return patient_dirs


def split_patient_ids(patient_ids, train_ratio=0.7, val_ratio=0.15):
    """
    Split patient IDs into training, validation, and testing sets.

    Args:
        patient_ids (list): List of patient IDs to be split.
        train_ratio (float): Proportion of patients to include in the training set. Default is 0.7.
        val_ratio (float): Proportion of patients to include in the validation set. Default is 0.15.

    Returns:
        tuple: Three lists - (train_ids, validation_ids, test_ids).
    """
    random.shuffle(patient_ids)
    train_end = int(len(patient_ids) * train_ratio)
    val_end = train_end + int(len(patient_ids) * val_ratio)
    return patient_ids[:train_end], patient_ids[train_end:val_end], patient_ids[val_end:]


def copy_files(patient_ids, source_dirs, dest_dir):
    """
    Copy patient subfolders to a new directory, skipping if the folder already exists.

    Args:
        patient_ids (list): List of patient IDs to copy.
        source_dirs (list or str): Source directory/directories to search for patient folders.
        dest_dir (str): Destination directory where patient folders will be copied.
    """
    if isinstance(source_dirs, str):
        source_dirs = [source_dirs]

    os.makedirs(dest_dir, exist_ok=True)

    copied_count = 0
    for patient_id in patient_ids:
        copied = False
        for source_dir in source_dirs:
            print(f"Checking source directory: {source_dir}")
            source_path = os.path.join(source_dir, patient_id)
            print(f"Source path: {source_path}")

            if os.path.exists(source_path):
                dest_path = os.path.join(dest_dir, patient_id)

                if not os.path.exists(dest_path):
                    shutil.copytree(source_path, dest_path)
                    copied = True
                    copied_count += 1
                    print(f"Copied {patient_id} to {dest_path}")
                else:
                    print(f"Skipping {patient_id}: Folder already exists in destination.")
                break

        if not copied:
            print(f"Warning: Patient folder {patient_id} does not exist in any source directory.")

    print(f"Copied {copied_count} patient folders to {dest_dir}.")


# Image Resizing and Augmentation Functions
def rotate_image(image_array, max_angle=30):
    angle = random.uniform(-max_angle, max_angle)
    return rotate(image_array, angle, reshape=False, mode='nearest')


def flip_image(image_array, horizontal=True):
    if horizontal:
        return np.fliplr(image_array)
    else:
        return np.flipud(image_array)


def adjust_brightness(image_array, factor=0.2):
    return np.clip(image_array + factor * np.max(image_array), 0, 255)


def adjust_contrast(image_array, factor=0.2):
    mean = np.mean(image_array)
    return np.clip((image_array - mean) * factor + mean, 0, 255)


def zoom_image(image_array, zoom_factor=1.2):
    return zoom(image_array, zoom_factor, order=3)


def augment_image(image_array):
    """
    Randomly apply one augmentation technique to the image array.

    Args:
        image_array (numpy.ndarray): Input image array.

    Returns:
        numpy.ndarray: Augmented image array.
    """
    augmentations = [rotate_image, flip_image, adjust_brightness, adjust_contrast, zoom_image]
    augmentation = random.choice(augmentations)

    if augmentation == flip_image:
        horizontal_flip = random.choice([True, False])
        return augmentation(image_array, horizontal=horizontal_flip)
    elif augmentation == rotate_image:
        return augmentation(image_array)
    else:
        return augmentation(image_array)


def resize_image(image_array, target_size):
    """
    Resize a given image array to the target size.

    Args:
        image_array (numpy.ndarray): Original image pixel array.
        target_size (tuple): Desired output size (height, width).

    Returns:
        numpy.ndarray: Resized image array.
    """
    zoom_factors = [target_size[0] / image_array.shape[0], target_size[1] / image_array.shape[1]]
    resized_image = zoom(image_array, zoom_factors, order=3)
    return resized_image


def resize_and_augment_image(image_array, target_size):
    augmented_image = augment_image(image_array)
    return resize_image(augmented_image, target_size)


def resize_and_augment_dataset(base_dir, target_size=(256, 256)):
    """
    Resize and augment images in the dataset.

    Applies augmentation only to training images, and resizes all images.

    Args:
        base_dir (str): Base directory containing train, validation, and test sets.
        target_size (tuple): Desired size (height, width).
    """
    for subset in ["train", "validation", "test"]:
        subset_dir = os.path.join(base_dir, subset)
        for root, _, files in os.walk(subset_dir):
            for file in files:
                if file.endswith(".dcm"):
                    dicom_path = os.path.join(root, file)
                    dicom = pydicom.dcmread(dicom_path)

                    image_array = dicom.pixel_array.astype(np.float32)

                    if subset == "train":
                        image_array = augment_image(image_array)

                    resized_array = resize_image(image_array, target_size)

                    dicom.PixelData = resized_array.astype(np.uint16).tobytes()
                    dicom.Rows, dicom.Columns = target_size

                    dicom.save_as(dicom_path)


# Main Dataset Management Functions
def reorganise_files():
    """
    Reorganise the dataset by splitting patient data into train, validation, and test sets.
    """
    print("Reorganising dataset...")

    all_patients = get_patient_ids(P10_DIR)
    train_ids, val_ids, test_ids = split_patient_ids(all_patients)

    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "validation")
    test_dir = os.path.join(DATASET_DIR, "test")

    print("Copying training data...")
    copy_files(train_ids, P10_DIR, train_dir)

    print("Copying validation data...")
    copy_files(val_ids, P10_DIR, val_dir)

    print("Copying test data...")
    copy_files(test_ids, P10_DIR, test_dir)

    print("Dataset reorganisation complete!")


def resize():
    """
    Resize the dataset images.
    """
    print("Resizing dataset...")
    resize_and_augment_dataset(DATASET_DIR, target_size=(256, 256))
    print("Resizing complete!")


# Main Entry Point
def main():
    reorganise_files()
    resize()


if __name__ == "__main__":
    main()
