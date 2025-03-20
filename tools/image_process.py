import os
from PIL import Image

def process_images(folder_path, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为图像文件
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            name_part = filename.split('.')[0]
            new_filename = f"{name_part}.jpg"
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    print(f"Processing {filename}: width={width}, height={height}")

                    # 计算图像中心0.8区域的坐标
                    new_width = int(width * 0.8)
                    new_height = int(height * 0.8)
                    left = (width - new_width) // 2
                    top = (height - new_height) // 2
                    right = left + new_width
                    bottom = top + new_height

                    # 裁剪图像
                    cropped_img = img.crop((left, top, right, bottom))

                    # 将裁剪后的图像保存到输出文件夹
                    output_path = os.path.join(output_folder, f"{new_filename}")
                    cropped_img.save(output_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # 输入文件夹路径
    # folder_path = '/home/narwal/mast3r-slam/MASt3R-SLAM/Narwal/datasets/84759'
    folder_path = '/home/narwal/MASt3R-SLAM/datasets/Narwal/84846'
    # 输出文件夹路径
    # output_folder = '/home/narwal/mast3r-slam/MASt3R-SLAM/Narwal/datasets/84759_new'
    output_folder = '/home/narwal/MASt3R-SLAM/datasets/Narwal/84846_new'
    process_images(folder_path, output_folder)

