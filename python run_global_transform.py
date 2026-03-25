import gradio as gr
import cv2
import numpy as np


# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])


def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):
    if image is None:
        return None

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)

    # Pad the image to avoid boundary issues (边缘填充，防止变换时图像被截断)
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size * 2 + image.shape[0], pad_size * 2 + image.shape[1], 3), dtype=np.uint8) + np.array(
        (255, 255, 255), dtype=np.uint8).reshape(1, 1, 3)
    image_new[pad_size:pad_size + image.shape[0], pad_size:pad_size + image.shape[1]] = image
    image = np.array(image_new)

    # 获取填充后图像的高(h)、宽(w)以及中心点坐标(cx, cy)
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2

    ### 核心部分：组合变换矩阵 (Apply Composition Transform) ###

    # 1. 缩放和旋转矩阵 (绕图像中心)
    # cv2.getRotationMatrix2D 返回一个 2x3 的矩阵
    rot_scale_mat = cv2.getRotationMatrix2D((cx, cy), rotation, scale)
    rot_scale_mat_3x3 = to_3x3(rot_scale_mat)  # 转换成 3x3 以便进行矩阵乘法

    # 2. 平移矩阵
    trans_mat_3x3 = np.array([[1, 0, translation_x],
                              [0, 1, translation_y], [0, 0, 1]
                              ], dtype=np.float32)

    # 3. 水平翻转矩阵
    flip_mat_3x3 = np.eye(3, dtype=np.float32)
    if flip_horizontal:
        # 绕中心水平翻转的数学等价操作：x_new = -x + w
        flip_mat_3x3[0, 0] = -1
        flip_mat_3x3[0, 2] = w

    # 4. 组合所有的变换矩阵 (注意矩阵乘法顺序: 先翻转 -> 再旋转缩放 -> 最后平移)
    composite_mat_3x3 = trans_mat_3x3 @ rot_scale_mat_3x3 @ flip_mat_3x3

    # 5. 提取 2x3 矩阵供 cv2.warpAffine 使用
    final_mat_2x3 = composite_mat_3x3[:2, :]

    # 6. 应用仿射变换，并用白色 (255, 255, 255) 填充空白边缘
    transformed_image = cv2.warpAffine(
        image,
        final_mat_2x3,
        (w, h),
        borderValue=(255, 255, 255)
    )

    return transformed_image


# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")

        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")

            # Right: Output image
            image_output = gr.Image(label="Transformed Image")

        # Automatically update the output when any slider or checkbox is changed
        inputs =[
            image_input, scale, rotation,
            translation_x, translation_y,
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
if __name__ == "__main__":
    interactive_transform().launch()
