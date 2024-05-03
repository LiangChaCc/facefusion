import cv2
import numpy as np
import onnxruntime
from utils import warp_face_by_translation, convert_face_landmark_68_to_5

class face_68_landmarks:
    def __init__(self, modelpath):
        """
        初始化face_68_landmarks对象。

        参数:
            modelpath (str): ONNX模型文件路径。
        """
        # 初始化模型
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        # 加载ONNX模型
        ###opencv-dnn读取onnx失败
        self.session = onnxruntime.InferenceSession(modelpath, sess_options=session_option)
        model_inputs = self.session.get_inputs()
        # 获取输入名称
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        # 获取输入形状
        self.input_shape = model_inputs[0].shape
        # 输入图像高度
        self.input_height = int(self.input_shape[2])
        # 输入图像宽度
        self.input_width = int(self.input_shape[3])


    def preprocess(self, srcimg, bounding_box):
        """
        图像预处理方法，用于将输入图像准备成模型所需的格式。
        对输入图像中人脸的裁剪和仿射变换操作，以便将人脸对齐并准备好用于模型推理
        参数:
            srcimg (numpy.ndarray): 输入图像，为BGR格式的numpy数组。
            bounding_box (numpy.ndarray): 检测到的人脸边界框坐标，格式为[xmin, ymin, xmax, ymax]。

        返回值:
            tuple: 包含经过预处理后的图像和仿射变换矩阵的元组。
        """
        # 计算裁剪比例
        # 通过np.subtract计算出bounding_box的宽度和高度。通过.max()方法找到宽度和高度中的最大值。使用195除以最大值，得到裁剪比例
        scale = 195 / np.subtract(bounding_box[2:], bounding_box[:2]).max()
        # 计算平移量 通过np.add计算出bounding_box的对角线的两个点的坐标之和 将结果乘以裁剪比例和0.5 将256减去上述结果，得到平移量
        translation = (256 - np.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
        # 调用warp_face_by_translation方法 该方法返回裁剪后的人脸图像crop_img和仿射变换矩阵affine_matrix
        crop_img, affine_matrix = warp_face_by_translation(srcimg, translation, scale, (256, 256))
        # 将裁剪后的人脸图像转换为模型输入格式 进行通道转换，使其通道维度置于第一维
        # 将图像的数据类型转换为np.float32，并将像素值范围归一化到[0, 1]。
        # 将crop_img的维度增加一个新的维度，以匹配模型的输入要求
        crop_img = crop_img.transpose(2, 0, 1).astype(np.float32) / 255.0
        crop_img = crop_img[np.newaxis, :, :, :]
        return crop_img, affine_matrix

    def detect(self, srcimg, bounding_box):
        """
        人脸关键点检测方法，用于在输入图像中检测人脸关键点。
        如果直接crop+resize,最后返回的人脸关键点有偏差
        参数:
            srcimg (numpy.ndarray): 输入图像，为BGR格式的numpy数组。
            bounding_box (numpy.ndarray): 检测到的人脸边界框坐标，格式为[xmin, ymin, xmax, ymax]。

        返回值:
           tuple: 包含检测到的68个人脸关键点和5个关键点的元组。
        """
        # 通过调用self.preprocess方法，对输入图像srcimg进行预处理，得到裁剪后的人脸图像input_tensor和仿射变换矩阵affine_matrix
        input_tensor, affine_matrix = self.preprocess(srcimg, bounding_box)

        # 执行推理
        # 使用onnxruntime的run方法执行模型推理，将input_tensor作为模型的输入
        # 获取模型的输出face_landmark_68，其中包含了检测到的68个人脸关键点的坐标
        face_landmark_68 = self.session.run(None, {self.input_names[0]: input_tensor})[0]
        # 对face_landmark_68进行处理，将其除以64，并乘以256，以将坐标缩放回原始图像尺寸
        face_landmark_68 = face_landmark_68[:, :, :2][0] / 64
        face_landmark_68 = face_landmark_68.reshape(1, -1, 2) * 256
        # 使用cv2.transform方法，将缩放后的关键点坐标进行仿射变换，以校正关键点的位置，使其对应到原始图像中。
        face_landmark_68 = cv2.transform(face_landmark_68, cv2.invertAffineTransform(affine_matrix))
        face_landmark_68 = face_landmark_68.reshape(-1, 2)
        # 将face_landmark_68转换为5个关键点的坐标，通过调用convert_face_landmark_68_to_5方法
        face_landmark_5of68 = convert_face_landmark_68_to_5(face_landmark_68)
        # 返回处理后的68个人脸关键点坐标face_landmark_68和5个关键点的坐标
        return face_landmark_68, face_landmark_5of68

if __name__ == '__main__':
    imgpath = 'images/5.jpg'
    srcimg = cv2.imread(imgpath)
    bounding_box = np.array([487, 236, 784, 624])
    
    # 初始化face_68landmarks检测器
    mynet = face_68_landmarks("weights/2dfan4.onnx")

    face_landmark_68, face_landmark_5of68 = mynet.detect(srcimg, bounding_box)

    # 绘制检测结果
    for i in range(face_landmark_68.shape[0]):
        cv2.circle(srcimg, (int(face_landmark_68[i,0]), int(face_landmark_68[i,1])), 3, (0, 255, 0), thickness=-1)
    cv2.imwrite('detect_face_68lanmarks.jpg', srcimg)
    winName = 'Deep learning face_68landmarks detection in ONNXRuntime'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
