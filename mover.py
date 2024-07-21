import os
import shutil

def move_images(source_directory, destination_directory, image_extensions):
    # Ensure source directory exists
    if not os.path.exists(source_directory):
        print(f"Source directory '{source_directory}' does not exist.")
        return
    
    # Ensure destination directory exists; create it if not
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    
    # Get a list of files in the source directory
    files = os.listdir(source_directory)
    
    # Filter files with the specified image extensions
    image_files = [file for file in files if any(file.lower().endswith(ext) for ext in image_extensions)]
    
    # Move image files to the destination directory
    for image_file in image_files:
        source_path = os.path.join(source_directory, image_file)
        destination_path = os.path.join(destination_directory, image_file)
        shutil.move(source_path, destination_path)
        print(f"Moved: {source_path} to {destination_path}")

# Example usage
source_directory = r"C:\Users\folfe\Desktop\finallyear\banana\Goodbanana"
destination_directory = r"C:\Users\folfe\Desktop\finallyear\banana\trainimages"
image_extensions = [".jpg", ".jpeg", ".png"]

move_images(source_directory, destination_directory, image_extensions)
