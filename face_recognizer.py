import cv2
import numpy as np
import onnxruntime
from utils import warp_face_by_face_landmark_5

class face_recognize:
    def __init__(self, modelpath):
        """
        初始化 face_recognize 类。

        参数：
            modelpath (str): ONNX 模型文件的路径。
        """
        # 初始化模型
        # 创建 ONNX 运行时会话选项对象
        session_option = onnxruntime.SessionOptions()
        # 设置日志级别为3，即INFO
        session_option.log_severity_level = 3
        # 加载 ONNX 模型并创建会话
        self.session = onnxruntime.InferenceSession(modelpath, providers=['CPUExecutionProvider'])
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

    def preprocess(self, srcimg, face_landmark_5):
        """
        对图像进行预处理。

        参数：
            srcimg (numpy.ndarray): 输入图像，BGR 格式，范围 [0, 255]。
            face_landmark_5 (numpy.ndarray): 人脸的五个关键点坐标，形状为 (5, 2)。

        返回值：
            numpy.ndarray: 预处理后的图像，形状为 (1, C, H, W)，范围 [-1, 1]。
        """
        # 通过五个人脸关键点变换裁剪图像
        crop_img, _ = warp_face_by_face_landmark_5(srcimg, face_landmark_5, 'arcface_112_v2', (112, 112))
        # 图像标准化
        crop_img = crop_img / 127.5 - 1
        # 转换图像通道顺序并转换数据类型
        crop_img = crop_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        # 扩展维度
        crop_img = np.expand_dims(crop_img, axis = 0)
        return crop_img

    def detect(self, srcimg, face_landmark_5):
        """
        在图像上执行人脸识别。

        参数：
            srcimg (numpy.ndarray): 输入图像，BGR 格式，范围 [0, 255]。
            face_landmark_5 (numpy.ndarray): 人脸的五个关键点坐标，形状为 (5, 2)。

        返回值：
            tuple: 包含两个元素的元组，第一个元素是原始嵌入向量，第二个元素是归一化后的嵌入向量。
        """
        # 预处理图像
        input_tensor = self.preprocess(srcimg, face_landmark_5)

        # 在图像上执行推理
        embedding = self.session.run(None, {self.input_names[0]: input_tensor})[0]
        # 拉平嵌入向量
        embedding = embedding.ravel()
        # 对嵌入向量进行归一化
        normed_embedding = embedding / np.linalg.norm(embedding)
        return embedding, normed_embedding
    
if __name__ == '__main__':
    imgpath = 'images/4.jpg'
    srcimg = cv2.imread(imgpath)
    face_landmark_5 = np.array([[568.2485,  398.9512 ],
                            [701.7346,  399.64795],
                            [634.2213,  482.92694],
                            [583.5656,  543.10187],
                            [684.52405, 543.125  ]])
    
    mynet = face_recognize('weights/arcface_w600k_r50.onnx')
    embedding, normed_embedding = mynet.detect(srcimg, face_landmark_5)
    print(embedding.shape, normed_embedding.shape)