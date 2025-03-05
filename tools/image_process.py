import os
from PIL import Image

def process_images(folder_path, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            name_part = filename.split('_')[0]
            new_filename = f"{name_part}.jpg"
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    print(f"Processing {filename}: width={width}, height={height}")

                    # Calculate the coordinates for the 0.7 region centered on the image
                    new_width = int(width * 0.8)
                    new_height = int(height * 0.8)
                    left = (width - new_width) // 2
                    top = (height - new_height) // 2
                    right = left + new_width
                    bottom = top + new_height

                    # Crop the image
                    cropped_img = img.crop((left, top, right, bottom))

                    # Save the cropped image to the output folder
                    output_path = os.path.join(output_folder, f"{new_filename}")
                    cropped_img.save(output_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    folder_path = '/home/narwal/mast3r-slam/MASt3R-SLAM/datasets/84759'
    output_folder = '/home/narwal/mast3r-slam/MASt3R-SLAM/datasets/84759_new'
    process_images(folder_path, output_folder)

