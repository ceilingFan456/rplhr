import os
import SimpleITK as sitk
import numpy as np
from PIL import Image

def read_and_save_nii_gz(input_path, output_dir):
    # Load the .nii.gz image
    image = sitk.ReadImage(input_path)
    image_array = sitk.GetArrayFromImage(image)  # Convert to numpy array

    # Check the range of the image and scale if necessary
    min_val, max_val = image_array.min(), image_array.max()
    print(f"Image range: {min_val} to {max_val}")

    if min_val >= 0 and max_val <= 1:
        image_array = (image_array * 255).astype(np.uint8)  # Scale to 0-255

    # Create output directory with the same name as the input file (without extension)
    base_name = os.path.basename(input_path).replace('.nii.gz', '')
    output_path = os.path.join(output_dir, base_name)
    os.makedirs(output_path, exist_ok=True)

    # Iterate through the image slices and save them as PNG files
    for i, slice in enumerate(image_array):
        slice_image = Image.fromarray(slice)
        slice_image = slice_image.convert("L")  # Convert to grayscale if necessary
        output_file = os.path.join(output_path, f"slice_{i:03d}.png")
        slice_image.save(output_file)
        print(f"Saved: {output_file}")

# Example usage
input_path = "/home/simtech/Qiming/RPLHR-CT/test_output/kits_x2/TVSRN/case_00010_pre.nii.gz"  # Path to the .nii.gz file
output_dir = "/home/simtech/Qiming/RPLHR-CT/test_output/kits_x2/TVSRN/case_00010_pre"  # Directory to save the image slices

# Process the .nii.gz file
read_and_save_nii_gz(input_path, output_dir)
