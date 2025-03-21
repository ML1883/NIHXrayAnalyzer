import os
import shutil
import polars as pl

def organize_images(csv_path, image_folder, output_folder, image_column="Image Index", label_column="Finding Labels"):
    """
    Organizes images from the given folder into subfolders based on diagnoses found in the CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing image indices and corresponding labels.
        image_folder (str): Folder where the images are currently stored.
        output_folder (str): Folder where the sorted images should be saved, organized by diagnosis.
        image_column (str): The name of the column in the CSV file that contains image indices (default is "Image Index").
        label_column (str): The name of the column in the CSV file that contains the diagnoses (default is "Finding Labels").
    
    Returns:
        None
    """
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read CSV file using Polars
    df = pl.read_csv(csv_path)

    # Check if output folder is empty before sorting images
    if os.path.exists(output_folder) and not os.listdir(output_folder):
        for row in df.iter_rows(named=True):
            image_name = row[image_column]
            diagnoses = row[label_column].split('|')  # Split diagnoses by '|'
            src_path = os.path.join(image_folder, image_name)

            # If the image exists, organize it into the appropriate diagnosis folder
            if os.path.exists(src_path):
                for diagnose in diagnoses:
                    diagnose = diagnose.strip()  # Remove any leading/trailing spaces
                    class_folder = os.path.join(output_folder, diagnose)
                    if not os.path.exists(class_folder):
                        os.makedirs(class_folder)

                    dest_path = os.path.join(class_folder, image_name)

                    shutil.copy(src_path, dest_path)
                    # Uncomment to print progress
                    # print(f"Copied {image_name} → {class_folder}")
            else:
                print(f"Warning: {image_name} not found in {image_folder}")
        
        print("Image sorting completed!")
    else:
        print("Sorted images folder already has content in it, skipping the sorting step.")
    
    # Print the number of files in each diagnosis folder
    for subobject in os.listdir(output_folder):
        subobject_path = os.path.join(output_folder, subobject)

        if os.path.isdir(subobject_path):
            num_files = len([f for f in os.listdir(subobject_path) if os.path.isfile(os.path.join(subobject_path, f))])
            print(f"Diagnosis {subobject}: {num_files} Xray(s)")