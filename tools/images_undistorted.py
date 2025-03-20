import cv2
import numpy as np
import os

# 相机内参矩阵
camera_matrix = np.array([
    [341.0743540942248, 0, 628.2818748026795],
    [0, 341.2789332650714, 361.5274708113607],
    [0, 0, 1]
])

# 畸变系数数组
dist_coeffs = np.array([
    0.3144895201419329,
    0.04997405492745952,
    0.00041306374668976403,
    -3.836312161525926e-05,
    -0.004222744688902338,
    0.3305490103821093,
    0.10278948843743044,
    -0.006141222184533711,
    -0.0011485409099832489,
    0.00017943253145389,
    -0.0008508668348584793,
    1.3992064115656973e-06
])

# 输入和输出文件夹路径
input_folder = '/home/narwal/MASt3R-SLAM/datasets/Narwal/84846'  # 替换为实际的输入文件夹路径
output_folder = '/home/narwal/MASt3R-SLAM/datasets/Narwal/84846_undistorted'  # 替换为实际的输出文件夹路径

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有图像文件
image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

for image_file in image_files:
    # 读取图像
    img = cv2.imread(os.path.join(input_folder, image_file))
    
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
    
    # 保存去畸变后的图像
    cv2.imwrite(os.path.join(output_folder, image_file), undistorted_img)
print("去畸变处理完成！")