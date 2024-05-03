import numpy as np
import onnxruntime
from utils import warp_face_by_face_landmark_5, create_static_box_mask, paste_back

# 设置人脸掩码模糊程度
FACE_MASK_BLUR = 0.3
# 设置人脸掩码的填充值，(top, right, bottom, left)，默认为0，表示不进行填充
FACE_MASK_PADDING = (0, 0, 0, 0)
# 设置 INSWrapper 128 模型的均值，用于归一化处理
INSWAPPER_128_MODEL_MEAN = [0.0, 0.0, 0.0]
# 设置 INSWrapper 128 模型的标准差，用于归一化处理
INSWAPPER_128_MODEL_STD = [1.0, 1.0, 1.0]

class swap_face:
    def __init__(self, modelpath):
        """
        初始化交换人脸对象检测器。

        参数：
        - modelpath：模型文件的路径。

        返回值：
        无。
        """
        # 初始化模型
        # 创建一个 ONNX 运行时的会话选项对象
        session_option = onnxruntime.SessionOptions()
        # 设置日志记录级别为 3，仅记录 ERROR 级别的日志信息
        session_option.log_severity_level = 3

        # 使用指定的模型路径和会话选项初始化一个 ONNX 推理会话对象 self.session
        self.session = onnxruntime.InferenceSession(modelpath, sess_options=session_option)  ###opencv-dnn读取onnx失败
        # 从 ONNX 模型中获取输入节点的信息
        model_inputs = self.session.get_inputs()
        # 将输入节点的名称存储在列表中，用于后续操作
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        # 获取输入节点的形状信息，并存储在 self.input_shape 中
        self.input_shape = model_inputs[0].shape
        # 获取输入节点的高度信息，并转换为整数类型
        self.input_height = int(self.input_shape[2])
        # 获取输入节点的宽度信息，并转换为整数类型
        self.input_width = int(self.input_shape[3])
        # 从本地加载模型矩阵，用于后续的处理
        self.model_matrix = np.load('model_matrix.npy')

    def process(self, target_img, source_face_embedding, target_landmark_5):
        """
        处理人脸交换。

        参数：
        - target_img：目标图像，即待处理的图像。
        - source_face_embedding：源人脸嵌入特征。
        - target_landmark_5：目标人脸的关键点坐标，包含左眼、右眼、鼻子和两侧嘴角。

        返回值：
        - dstimg：处理后的图像，即进行人脸交换后的结果图像。
        """
        # 对目标图像进行预处理，包括裁剪人脸区域、生成人脸掩码、归一化处理等
        crop_img, affine_matrix = warp_face_by_face_landmark_5(target_img, target_landmark_5,
                                                               'arcface_128_v2', (128, 128))
        # 初始化裁剪后的人脸掩码列表
        crop_mask_list = []

        # 生成静态方框掩码并添加到掩码列表中
        box_mask = create_static_box_mask((crop_img.shape[1],crop_img.shape[0]),
                                          FACE_MASK_BLUR, FACE_MASK_PADDING)
        crop_mask_list.append(box_mask)
        # 将裁剪后的图像转换为指定格式，并进行归一化处理
        crop_img = crop_img[:, :, ::-1].astype(np.float32) / 255.0
        crop_img = (crop_img - INSWAPPER_128_MODEL_MEAN) / INSWAPPER_128_MODEL_STD
        crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), axis = 0).astype(np.float32)

        # 将源人脸嵌入向量重塑为二维数组，并通过模型矩阵进行线性变换和归一化处理
        source_embedding = source_face_embedding.reshape((1, -1))
        source_embedding = np.dot(source_embedding, self.model_matrix) / np.linalg.norm(source_embedding)

        # 对图像进行推理处理，获取处理后的结果
        result = self.session.run(None, {'target':crop_img, 'source':source_embedding})[0][0]

        # 对推理结果进行归一化处理
        result = result.transpose(1, 2, 0)
        result = (result * 255.0).round()
        result = result[:, :, ::-1]

        # 获取裁剪后的掩码并取列表中的最小值，并对其进行截取处理，得到最终的裁剪掩码
        crop_mask = np.minimum.reduce(crop_mask_list).clip(0, 1)
        # 将处理后的结果与裁剪掩码进行合并，得到最终的图像结果
        dstimg = paste_back(target_img, result, crop_mask, affine_matrix)
        # 返回处理后的图像
        return dstimg