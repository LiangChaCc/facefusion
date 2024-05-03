import numpy as np
import onnxruntime
from utils import warp_face_by_face_landmark_5, create_static_box_mask, paste_back, blend_frame

# 设置人脸掩码模糊程度
FACE_MASK_BLUR = 0.3
# 设置人脸掩码的填充值，(top, right, bottom, left)，默认为0，表示不进行填充
FACE_MASK_PADDING = (0, 0, 0, 0)

class enhance_face:
    def __init__(self, modelpath):
        """
        初始化 enhance_face 类。

        参数：
            modelpath (str): ONNX 模型文件的路径。
        """
        # 创建 ONNX 运行时会话选项对象
        session_option = onnxruntime.SessionOptions()
        # 设置日志级别为3，即INFO
        session_option.log_severity_level = 3
        # 加载 ONNX 模型并创建会话
        self.session = onnxruntime.InferenceSession(modelpath, sess_options=session_option)  ###opencv-dnn读取onnx失败
        # 获取模型输入信息
        model_inputs = self.session.get_inputs()
        # 获取输入名称列表
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        # 获取输入形状信息
        self.input_shape = model_inputs[0].shape
        # 获取输入高度
        self.input_height = int(self.input_shape[2])
        # 获取输入宽度
        self.input_width = int(self.input_shape[3])

    def process(self, target_img, target_landmark_5):
        """
        对目标图像进行人脸增强处理。

        参数：
            target_img (numpy.ndarray): 目标图像，RGB 格式，范围 [0, 255]。
            target_landmark_5 (numpy.ndarray): 目标人脸的五个关键点坐标，形状为 (5, 2)。

        返回值：
            numpy.ndarray: 处理后的图像，RGB 格式，范围 [0, 255]。
        """
        # 通过五个人脸关键点变换裁剪图像
        crop_img, affine_matrix = warp_face_by_face_landmark_5(target_img, target_landmark_5,
                                                               'ffhq_512', (512, 512))
        # 创建静态盒子掩码
        box_mask = create_static_box_mask((crop_img.shape[1],crop_img.shape[0]),
                                          FACE_MASK_BLUR, FACE_MASK_PADDING)
        crop_mask_list = [box_mask]
        # 图像预处理
        # RGB转换为BGR，并将像素值范围归一化到 [0, 1]
        crop_img = crop_img[:, :, ::-1].astype(np.float32) / 255.0
        # 图像标准化
        crop_img = (crop_img - 0.5) / 0.5
        # 转换为 ONNX 模型的输入格式
        crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), axis = 0).astype(np.float32)

        # 在图像上执行推理
        result = self.session.run(None, {'input':crop_img})[0][0]
        # 规范化裁剪帧
        # 将像素值限制在 [-1, 1] 范围内
        result = np.clip(result, -1, 1)
        # 将像素值范围调整到 [0, 1]
        result = (result + 1) / 2
        # 转置通道顺序
        result = result.transpose(1, 2, 0)
        # 将像素值转换为整数
        result = (result * 255.0).round()
        # 将像素值范围调整到 [0, 255] 并转换为 RGB 格式
        result = result.astype(np.uint8)[:, :, ::-1]
        # 求取掩码列表中的最小值，并将像素值限制在 [0, 1] 范围内
        crop_mask = np.minimum.reduce(crop_mask_list).clip(0, 1)
        # 将裁剪的图像粘贴回原始图像中
        paste_frame = paste_back(target_img, result, crop_mask, affine_matrix)
        # 将粘贴的图像与原始图像进行混合处理
        dstimg = blend_frame(target_img, paste_frame)
        return dstimg