import numpy as np 
import cv2

def warp_face_by_translation(temp_img, translation, scale, crop_size):
    """
    将输入图像根据给定的平移和缩放参数进行仿射变换。

    参数：
    - temp_img：输入图像，即待处理的原始图像。
    - translation：平移参数，表示图像的平移量，形如 (tx, ty)。
    - scale：缩放参数，表示图像的缩放比例。
    - crop_size：裁剪尺寸，表示输出图像的尺寸，形如 (width, height)。

    返回值：
    - crop_img：变换后的图像，即根据平移和缩放参数进行了仿射变换后的图像。
    - affine_matrix：变换矩阵，即用于对输入图像进行仿射变换的变换矩阵。

    """
    # 构建仿射矩阵
    affine_matrix = np.array([[scale, 0, translation[0]], [0, scale, translation[1]]])
    # 对输入图像进行仿射变换
    crop_img = cv2.warpAffine(temp_img, affine_matrix, crop_size)
    return crop_img, affine_matrix

def convert_face_landmark_68_to_5(landmark_68):
    """
    将 68 个人脸关键点坐标转换为 5 个关键点坐标。

    参数：
    - landmark_68：68 个人脸关键点的坐标，形如 [(x1, y1), (x2, y2), ..., (x68, y68)]。

    返回值：
    - face_landmark_5：5 个关键点的坐标，形如 [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]。

    """
    # 计算左眼、右眼、鼻子、左嘴角、右嘴角关键点的坐标
    left_eye = np.mean(landmark_68[36:42], axis = 0)
    right_eye = np.mean(landmark_68[42:48], axis = 0)
    nose = landmark_68[30]
    left_mouth_end = landmark_68[48]
    right_mouth_end = landmark_68[54]
    # 将计算得到的关键点坐标组合成 5 个关键点的坐标
    face_landmark_5 = np.array([left_eye, right_eye, nose, left_mouth_end, right_mouth_end])
    return face_landmark_5

# 定义了三种不同尺寸的人脸模板，用于人脸关键点的标准化处理
# 在人脸对齐和人脸裁剪过程中，作为目标形状的参考
# 通过与人脸关键点的位置进行比较，可以进行相应的仿射变换，将人脸对齐到指定的模板形状，从而实现人脸的标准化处理。
# 有助于提高人脸识别、人脸验证等任务的准确性和稳定性。
TEMPLATES = {'arcface_112_v2': np.array([[ 0.34191607, 0.46157411 ],
                                         [ 0.65653393, 0.45983393 ],
                                         [ 0.50022500, 0.64050536 ],
                                         [ 0.37097589, 0.82469196 ],
                                         [ 0.63151696, 0.82325089 ]]),
             'arcface_128_v2': np.array([[ 0.36167656, 0.40387734 ],
                                         [ 0.63696719, 0.40235469 ],
                                         [ 0.50019687, 0.56044219 ],
                                         [ 0.38710391, 0.72160547 ],
                                         [ 0.61507734, 0.72034453 ]]),
             'ffhq_512': np.array([[ 0.37691676, 0.46864664 ],
                                   [ 0.62285697, 0.46912813 ],
                                   [ 0.50123859, 0.61331904 ],
                                   [ 0.39308822, 0.72541100 ],
                                   [ 0.61150205, 0.72490465 ]])}

def warp_face_by_face_landmark_5(temp_vision_frame, face_landmark_5, template, crop_size):
    """
    根据 5 个关键点的坐标进行仿射变换，将人脸对齐到预定义的模板形状。

    参数：
    - temp_vision_frame：输入图像，即待处理的原始图像。
    - face_landmark_5：5 个关键点的坐标，形如 [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]。
    - template：预定义的模板形状，用于对齐人脸的目标形状。
    - crop_size：裁剪尺寸，表示输出图像的尺寸，形如 (width, height)。

    返回值：
    - crop_img：变换后的图像，即根据 5 个关键点的坐标进行了仿射变换后的图像。
    - affine_matrix：变换矩阵，即用于对输入图像进行仿射变换的变换矩阵。

    """
    # 根据预定义模板的尺寸对模板进行归一化处理
    normed_template = TEMPLATES.get(template) * crop_size
    # 使用 RANSAC 算法估计仿射变换矩阵
    affine_matrix = cv2.estimateAffinePartial2D(face_landmark_5, normed_template, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
    # 对输入图像进行仿射变换
    crop_img = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size, borderMode = cv2.BORDER_REPLICATE, flags = cv2.INTER_AREA)
    return crop_img, affine_matrix

def create_static_box_mask(crop_size, face_mask_blur, face_mask_padding):
    """
    创建一个静态的方框遮罩，用于控制人脸边界的模糊程度和填充百分比。

    参数：
    - crop_size：裁剪尺寸，表示输出图像的尺寸，形如 (width, height)。
    - face_mask_blur：人脸边界模糊程度，即模糊半径的百分比。
    - face_mask_padding：人脸填充百分比，表示人脸边界的填充程度。

    返回值：
    - box_mask：静态的方框遮罩，用于控制人脸边界的模糊程度和填充百分比。

    """
    # 计算模糊半径和填充百分比所对应的参数值
    blur_amount = int(crop_size[0] * 0.5 * face_mask_blur)
    blur_area = max(blur_amount // 2, 1)
    # 创建初始的遮罩图像，全部初始化为 1
    box_mask = np.ones(crop_size, np.float32)
    # 设置人脸边界的填充区域为 0
    box_mask[:max(blur_area, int(crop_size[1] * face_mask_padding[0] / 100)), :] = 0
    box_mask[-max(blur_area, int(crop_size[1] * face_mask_padding[2] / 100)):, :] = 0
    box_mask[:, :max(blur_area, int(crop_size[0] * face_mask_padding[3] / 100))] = 0
    box_mask[:, -max(blur_area, int(crop_size[0] * face_mask_padding[1] / 100)):] = 0
    # 如果模糊半径大于 0，则对遮罩图像进行高斯模糊处理
    if blur_amount > 0:
        box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
    return box_mask

def paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix):
    """
    将裁剪后的人脸图像粘贴回原始图像中，并进行透明度融合。

    参数：
    - temp_vision_frame：输入图像，即原始图像。
    - crop_vision_frame：裁剪后的人脸图像。
    - crop_mask：裁剪图像的遮罩，用于控制裁剪区域的透明度。
    - affine_matrix：仿射变换矩阵，用于将裁剪图像粘贴回原始图像中。

    返回值：
    - paste_vision_frame：粘贴后的图像，即将裁剪后的人脸图像粘贴回原始图像中，并进行透明度融合后的图像。

    """
    # 计算逆变换矩阵
    inverse_matrix = cv2.invertAffineTransform(affine_matrix)
    temp_size = temp_vision_frame.shape[:2][::-1]
    # 使用逆变换矩阵将裁剪图像的遮罩变换到原始图像空间中
    inverse_mask = cv2.warpAffine(crop_mask, inverse_matrix, temp_size).clip(0, 1)
    # 使用逆变换矩阵将裁剪图像变换到原始图像空间中
    inverse_vision_frame = cv2.warpAffine(crop_vision_frame, inverse_matrix,
                                          temp_size, borderMode = cv2.BORDER_REPLICATE)
    # 将裁剪后的人脸图像粘贴回原始图像中，并进行透明度融合
    paste_vision_frame = temp_vision_frame.copy()
    paste_vision_frame[:, :, 0] = inverse_mask * inverse_vision_frame[:, :, 0] + (1 - inverse_mask) * temp_vision_frame[:, :, 0]
    paste_vision_frame[:, :, 1] = inverse_mask * inverse_vision_frame[:, :, 1] + (1 - inverse_mask) * temp_vision_frame[:, :, 1]
    paste_vision_frame[:, :, 2] = inverse_mask * inverse_vision_frame[:, :, 2] + (1 - inverse_mask) * temp_vision_frame[:, :, 2]
    return paste_vision_frame

def blend_frame(temp_vision_frame, paste_vision_frame, FACE_ENHANCER_BLEND=80):
    """
    将裁剪的图像和原始图像进行混合，增强人脸区域的细节。

    参数：
    - temp_vision_frame：输入图像，即原始图像。
    - paste_vision_frame：裁剪后的人脸图像。
    - FACE_ENHANCER_BLEND：混合比例，用于控制混合的程度。

    返回值：
    - temp_vision_frame：混合后的图像，即将裁剪的图像和原始图像进行混合，增强人脸区域的细节后的图像。

    """
    # 计算混合比例
    face_enhancer_blend = 1 - (FACE_ENHANCER_BLEND / 100)
    # 将裁剪的图像和原始图像进行混合
    temp_vision_frame = cv2.addWeighted(temp_vision_frame, face_enhancer_blend, paste_vision_frame, 1 - face_enhancer_blend, 0)
    return temp_vision_frame