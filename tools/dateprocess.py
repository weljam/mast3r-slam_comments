import os
import cv2
" change the hz "

def reduce_frame_rate(input_dir, output_dir, original_fps=30, target_fps=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_interval = original_fps / target_fps
    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    for i, frame_file in enumerate(frame_files):
        if i % frame_interval < 1:
            img = cv2.imread(os.path.join(input_dir, frame_file))
            cv2.imwrite(os.path.join(output_dir, frame_file), img)

if __name__ == "__main__":
    input_directory = '/home/narwal/mast3r-slam/MASt3R-SLAM/datasets/diff_HZ/tum/f1_floor'
    output_directory = '/home/narwal/mast3r-slam/MASt3R-SLAM/datasets/diff_HZ/tum/f1_floor_5HZ'
    reduce_frame_rate(input_directory, output_directory)