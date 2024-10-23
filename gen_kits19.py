import os
import shutil
import random
import argparse
import SimpleITK as sitk
import numpy as np
from pathlib import Path

# Global variables for counts
TRAIN_COUNT = 50  # Hard-coded count for training set
VAL_COUNT = 10    # Hard-coded count for validation set

# Manually specified list of evaluation cases
EVAL_CASES = [
    'case_00010',
    'case_00045',
    'case_00052',
    'case_00089',
    'case_00120',
    'case_00135',
    'case_00140',
    'case_00162',
    'case_00197',
    'case_00210',
    'case_00230',
    'case_00291',
    'case_00295',
]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Organize .nii.gz data into train, eval, and val splits with 1mm and 5mm resolutions.")
    parser.add_argument('--source_dir', type=str, default='/home/simtech/Qiming/kits19', help='Path to the source data directory (e.g., ./kits19/)')
    parser.add_argument('--dest_dir', type=str, required=True, help='Path to the destination directory where organized data will be stored.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--scale', type=int, default=1, help='Scale factor for the data.')
    return parser.parse_args()

def align(data):
    if data.shape[1] != data.shape[2]:
        if data.shape[0] == data.shape[2]:
            data = data.transpose(1, 0, 2)
        else:
            data = data.transpose(2, 1, 0)
    return data

def get_all_cases(source_dir):
    """
    Retrieves all case directories within the source directory.
    Assumes each case is in a subfolder named case_00xxx.
    """
    data_dir = os.path.join(source_dir, 'data')
    cases = [d.name for d in os.scandir(data_dir) if d.is_dir() and d.name.startswith('case_')]
    print(f"Found {len(cases)} cases in {data_dir}")
    return cases

def create_destination_folders(dest_dir, splits=['train', 'eval', 'val'], resolutions=['1mm', '5mm']):
    """
    Creates the necessary folder structure in the destination directory.
    """
    for split in splits:
        for res in resolutions:
            path = os.path.join(dest_dir, split, res)
            os.makedirs(path, exist_ok=True)

def process_and_save_nifti(source_path, dest_path_1mm, dest_path_5mm, scale):
    """
    Processes the .nii.gz file to extract 1mm and 5mm slices, normalize to [0, 1], and saves them using SimpleITK.
    """
    try:
        # Read the image using SimpleITK
        img = sitk.ReadImage(source_path)
        data = sitk.GetArrayFromImage(img)  # Shape: (D, H, W)
        data = align(data)
        # data = data[:100, :, :]  # Optional: Limit to first 100 slices
        affine = img.GetDirection()
        origin = img.GetOrigin()
        spacing = img.GetSpacing()

        D, H, W = data.shape
        print(f"Processing {os.path.basename(source_path)} with shape D={D}, H={H}, W={W}")

        # Normalize data to the 0-1 range
        min_value = np.min(data)
        max_value = np.max(data)
        if max_value > min_value:
            data = (data - min_value) / (max_value - min_value)
        else:
            print(f"Warning: max_value equals min_value for {source_path}, skipping normalization.")

        # 5mm Processing: Store slices where D % 5 == 0
        slices_5mm = data[::scale, :, :]  # Equivalent to D indices: 0,5,10,...
        if slices_5mm.size == 0:
            print(f"No slices found for 5mm in {os.path.basename(source_path)}. Skipping 5mm.")
        else:
            img_5mm = sitk.GetImageFromArray(slices_5mm)
            img_5mm.SetDirection(affine)
            img_5mm.SetOrigin(origin)
            img_5mm.SetSpacing(spacing)
            sitk.WriteImage(img_5mm, dest_path_5mm)
            print(f"Saved 5mm data to {dest_path_5mm}, shape={slices_5mm.shape}")

        # 1mm Processing: Use formula ((D//5)-1)*4 + D//5 slices
        d = slices_5mm.shape[0]
        slices_1mm = data[:(d - 1) * (scale-1) + d, :, :]

        if slices_1mm.size == 0:
            print(f"No slices found for 1mm in {os.path.basename(source_path)}. Skipping 1mm.")
        else:
            img_1mm = sitk.GetImageFromArray(slices_1mm)
            img_1mm.SetDirection(affine)
            img_1mm.SetOrigin(origin)
            img_1mm.SetSpacing(spacing)
            sitk.WriteImage(img_1mm, dest_path_1mm)
            print(f"Saved 1mm data to {dest_path_1mm}, shape={slices_1mm.shape}")
    except Exception as e:
        print(f"Error processing {source_path}: {e}")

def main():
    args = parse_arguments()

    source_dir = args.source_dir
    dest_dir = args.dest_dir
    seed = args.seed

    # Set random seed for reproducibility
    random.seed(seed)

    # Get all cases
    all_cases = get_all_cases(source_dir)
    print(f"Total available cases: {len(all_cases)}")

    # Ensure that the evaluation cases exist in all_cases
    for case in EVAL_CASES:
        if case not in all_cases:
            raise ValueError(f"Evaluation case {case} not found in the source directory.")

    # Remove evaluation cases from all_cases
    remaining_cases = [case for case in all_cases if case not in EVAL_CASES]

    # Shuffle the remaining cases
    random.shuffle(remaining_cases)

    # Check if there are enough cases for training and validation
    if len(remaining_cases) < (TRAIN_COUNT + VAL_COUNT):
        raise ValueError(f"Not enough cases to split into training and validation sets. Required: {TRAIN_COUNT + VAL_COUNT}, Available: {len(remaining_cases)}")

    # Split into training and validation sets
    train_cases = remaining_cases[:TRAIN_COUNT]
    val_cases = remaining_cases[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]

    print(f"Training cases: {len(train_cases)}")
    print(f"Validation cases: {len(val_cases)}")
    print(f"Evaluation cases: {len(EVAL_CASES)}")

    # Create destination folders
    create_destination_folders(dest_dir)
    print(f"Destination folders created at {dest_dir}")

    # Define splits
    splits = {
        'train': train_cases,
        'val': val_cases,
        'eval': EVAL_CASES
    }

    # Process each split
    for split_name, cases in splits.items():
        for case in cases:
            source_case_dir = os.path.join(source_dir, "data", case)
            source_nifti = os.path.join(source_case_dir, f"{case}.nii.gz")

            if not os.path.exists(source_nifti):
                print(f"Source file {source_nifti} does not exist. Skipping.")
                continue

            # Define destination paths
            dest_1mm = os.path.join(dest_dir, split_name, '1mm', f"{case}.nii.gz")
            dest_5mm = os.path.join(dest_dir, split_name, '5mm', f"{case}.nii.gz")

            # Process and save
            process_and_save_nifti(source_nifti, dest_1mm, dest_5mm, args.scale)

    print("Data organization complete.")

if __name__ == '__main__':
    main()


## python3 gen_kits19.py --dest_dir ./kits_data_x2 --scale 2