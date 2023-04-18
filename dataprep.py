import os
import shutil

# Define the source directory and the target directories
source_dir = "train"
input_dir = "input"
target_dir = "target"

# Create the target directories if they don't exist
if not os.path.exists(input_dir):
    os.makedirs(input_dir)
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Loop over all files in the source directory
for filename in os.listdir(source_dir):
    # Check if the file name contains "_sat" or "_mask"
    if "_sat" in filename:
        # Move the file to the input directory
        shutil.move(os.path.join(source_dir, filename),
                    os.path.join(input_dir, filename.replace("_sat", "")))
        print(f"Moved {filename} to {input_dir}")
    elif "_mask" in filename:
        # Move the file to the target directory
        shutil.move(os.path.join(source_dir, filename),
                    os.path.join(target_dir, filename.replace("_mask", "")))
        print(f"Moved {filename} to {target_dir}")
    else:
        # Ignore the file
        print(f"Ignored {filename}")
