import os
import shutil
import polars as pl

def organize_images_by_label_folder(csv_path, image_folder, output_folder, image_column="Image Index", label_column="Finding Labels"):
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


def sort_images_train_test(source_folder=os.path.join("..", "data", "images_raw", "images"), train_folder=os.path.join("..", "data", "images_train"), test_folder=os.path.join("..", "data", "images_test"), N=10000, train_ratio=0.7, delete_test_train_folder=False):
    """
    Sorts image files from the source folder into train and test folders.

    Args:
        source_folder (str): Path to the folder containing the images.
        train_folder (str): Path to the train folder.
        test_folder (str): Path to the test folder.
        N (int): Number of images to process, sorted by name in ascending order.
        train_ratio (float): Ratio of images to be placed in the train folder (default is 0.7).
        delete_test_trainer_folder (bool): Whether to first delete all image files in the test and train folder sorting.

    Returns:
        None
    """
    
    if delete_test_train_folder:
        for filename in os.listdir(train_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(train_folder, filename)
                try:
                    os.remove(file_path)
                    # print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

        for filename in os.listdir(test_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(test_folder, filename)
                try:
                    os.remove(file_path)
                    # print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

    # Make sure destinations folders are empty and excist. Otherwise skip
    if os.path.exists(train_folder) and os.path.exists(test_folder) and not os.listdir(train_folder) and not os.listdir(test_folder):
        
        # Get image files sorted by name
        image_files = sorted([f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Limit to N images
        selected_images = image_files[:N]
        
        # Compute split index
        # TODO: add some randemization here.
        train_count = int(len(selected_images) * train_ratio)
        
        # Move images to respective folders
        for i, image in enumerate(selected_images):
            src_path = os.path.join(source_folder, image)
            dest_folder = train_folder if i < train_count else test_folder
            dest_path = os.path.join(dest_folder, image)
            shutil.copy(src_path, dest_path)
            
        print(f"Sorted {len(selected_images)} images: {train_count} to train, {len(selected_images) - train_count} to test.")
    else:
        print("No files transferred since destiantion folders aren't empty or they simply don't excist.")
