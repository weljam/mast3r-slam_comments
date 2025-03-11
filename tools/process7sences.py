import os
import shutil

def extract_color_images(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.color.png'):
                source_file = os.path.join(root, file)
                new_file_name = file.replace('frame-', '').replace('.color', '')
                destination_file = os.path.join(destination_folder, new_file_name)
                shutil.copy2(source_file, destination_file)
                print(f"Copied {source_file} to {destination_file}")

# Example usage
source_folder = '/path/to/source/folder'
destination_folder = '/path/to/destination/folder'
extract_color_images(source_folder, destination_folder)
