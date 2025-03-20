import cv2

def resize_image(image_path, width, height):
    # 读取图像
    image = cv2.imread(image_path)
    print(image.shape)
    if image is None:
        raise ValueError(f"无法打开图像文件: {image_path}")

    # 调整图像大小
    resized_image = cv2.resize(image, (width, height))

    # 显示原始图像和调整后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Resized Image', resized_image)

    # 等待按键按下
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 按下 ESC 键退出
            break

    # 销毁所有窗口
    cv2.destroyAllWindows()

# 示例调用
if __name__ == "__main__":
    image_path = '../datasets/Narwal/84846_lower/000000001528_936817000.jpg'
    width = 224
    height = 224
    resize_image(image_path, width, height)