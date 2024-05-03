import cv2
import numpy as np
import onnxruntime
import argparse

'''
实现了基于YOLO模型的实时人脸检测功能。
1.YOLO（You Only Look Once）算法：YOLO是一种目标检测算法，通过将目标检测任务转化为单次神经网络推理过程来实现高效的实时目标检测。
2.ONNX（Open Neural Network Exchange）：ONNX是一种开放的神经网络模型交换格式，允许不同的深度学习框架之间共享训练的模型。
    这段代码中使用了ONNX模型来加载和运行YOLO模型。
3.OpenCV：OpenCV是一个广泛使用的计算机视觉库，提供了许多图像处理和计算机视觉算法的实现。
    在这段代码中，使用OpenCV进行图像的读取、预处理和绘制检测结果等操作。
4.NMS（Non-Maximum Suppression，非极大值抑制）：NMS是一种用于消除重叠较多的边界框的技术，它保留具有高置信度的边界框，
    并剔除与其重叠度较高的其他边界框。在这段代码中，使用了OpenCV的NMSBoxes函数来执行非最大值抑制。
5.神经网络预处理和后处理：在YOLOface_8n类中，有预处理方法和后处理方法。
    预处理方法用于将输入图像准备成模型所需的格式，后处理方法用于处理模型输出以得到最终的检测结果。
6.命令行参数解析：使用了Python的argparse库来解析命令行参数，
    使得可以在命令行中指定图像路径和置信度阈值等参数。
'''
class YOLOface_8n:
    def __init__(self, modelpath, conf_thres=0.5, iou_thresh=0.4):
        """
       初始化YOLOface_8n对象。

       参数:
           modelpath (str): ONNX模型文件路径。
           conf_thres (float): 置信度阈值，默认为0.5。
           iou_thresh (float): IoU（交并比）阈值，默认为0.4。
       """
        self.conf_threshold = conf_thres    # 置信度阈值
        self.iou_threshold = iou_thresh     # IoU（交并比）阈值
        # 初始化模型
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        # 加载ONNX模型
        self.session = onnxruntime.InferenceSession(modelpath, sess_options=session_option)  ###opencv-dnn读取onnx失败
        model_inputs = self.session.get_inputs()
        # 获取输入名称
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        # 获取输入形状
        self.input_shape = model_inputs[0].shape
        # 输入图像高度
        self.input_height = int(self.input_shape[2])
        # 输入图像宽度
        self.input_width = int(self.input_shape[3])

    def preprocess(self, srcimg):
        """
        图像预处理方法，用于将输入图像准备成模型所需的格式。

        参数:
            srcimg (numpy.ndarray): 输入图像，为BGR格式的numpy数组。

        返回值:
            numpy.ndarray: 经过预处理后的图像，用于输入模型进行推理。
        """
        # 获取原始图像的高度和宽度
        height, width = srcimg.shape[:2]
        temp_image = srcimg.copy()
        # 如果图像尺寸大于输入尺寸，则进行缩放
        if height > self.input_height or width > self.input_width:
            scale = min(self.input_height / height, self.input_width / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            temp_image = cv2.resize(srcimg, (new_width, new_height))
        # 高度缩放比例
        self.ratio_height = height / temp_image.shape[0]
        # 宽度缩放比例
        self.ratio_width = width / temp_image.shape[1]
        # 边界填充
        input_img = cv2.copyMakeBorder(temp_image, 0, self.input_height - temp_image.shape[0],
                                       0, self.input_width - temp_image.shape[1], cv2.BORDER_CONSTANT,value=0)
        # 将输入像素值缩放到0到1之间
        input_img = (input_img.astype(np.float32) - 127.5) / 128.0
        input_img = input_img.transpose(2, 0, 1)
        input_img = input_img[np.newaxis, :, :, :]
        return input_img

    def detect(self, srcimg):
        """
        物体检测方法，用于在输入图像中检测物体。

        参数:
            srcimg (numpy.ndarray): 输入图像，为BGR格式的numpy数组。

        返回值:
            tuple: 包含检测到的边界框、关键点和置信度的元组。
        """
        input_tensor = self.preprocess(srcimg)

        # 在图像上执行推理
        outputs = self.session.run(None, {self.input_names[0]: input_tensor})[0]
        boxes, kpts, scores = self.postprocess(outputs) # 后处理
        return boxes, kpts, scores

    def postprocess(self, outputs):
        """
       后处理方法，用于处理模型输出以得到最终的检测结果。

       参数:
           outputs (numpy.ndarray): 模型输出。

       返回值:
           tuple: 包含处理后的边界框、关键点和置信度的元组。
       """
        bounding_box_list, face_landmark5_list, score_list= [], [], []
        
        outputs = np.squeeze(outputs, axis=0).T
        bounding_box_raw, score_raw, face_landmark_5_raw = np.split(outputs, [ 4, 5 ], axis = 1)
        keep_indices = np.where(score_raw > self.conf_threshold)[0] # 根据置信度筛选框
        if keep_indices.any():
            bounding_box_raw, face_landmark_5_raw, score_raw = (bounding_box_raw[keep_indices],
                                                                face_landmark_5_raw[keep_indices],
                                                                score_raw[keep_indices])
            bboxes_wh = bounding_box_raw.copy()
            # 将中心坐标和宽高转换为左上角坐标和宽高
            bboxes_wh[:, :2] = bounding_box_raw[:, :2] - 0.5 * bounding_box_raw[:, 2:]
            bboxes_wh *= np.array([[self.ratio_width, self.ratio_height, self.ratio_width, self.ratio_height]])  ### 合理使用广播法则
            # 根据缩放比例调整坐标
            # 合理使用广播法则,每个点的信息是(x,y,conf), 第3个元素点的置信度，可以不要，那也就需要要乘以1
            face_landmark_5_raw *= (np.tile(np.array([self.ratio_width, self.ratio_height, 1]), 5)
                                    .reshape((1, 15)))
            score_raw = score_raw.flatten()

            indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), score_raw.tolist(), self.conf_threshold, self.iou_threshold)
            # 进行非最大抑制
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()
            if len(indices) > 0:
                # 将坐标转换为[x_min, y_min, x_max, y_max]格式
                bounding_box_list = list(map(lambda x:np.array([x[0], x[1], x[0]+x[2], x[1]+x[3]], dtype=np.float64), bboxes_wh[indices])) ###xywh转到xminyminxmaxymax
                score_list = list(score_raw[indices])
                face_landmark5_list = list(face_landmark_5_raw[indices])

        return bounding_box_list, face_landmark5_list, score_list

    def draw_detections(self, image, boxes,  kpts, scores):
        """
        绘制检测结果方法，用于在图像上绘制检测到的物体和关键点。

        参数:
           image (numpy.ndarray): 输入图像，为BGR格式的numpy数组。
           boxes (list): 包含检测到的边界框的列表。
           kpts (list): 包含检测到的关键点的列表。
           scores (list): 包含检测到的置信度的列表。

        返回值:
           numpy.ndarray: 经过绘制检测结果后的图像。
        """
        for box, kp, score in zip(boxes, kpts, scores):
            xmin, ymin, xmax, ymax = box.astype(int)
            
            # 绘制矩形框
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
            label = "face:"+str(round(score,2))
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
            for i in range(5):
                cv2.circle(image, (int(kp[i * 3]), int(kp[i * 3 + 1])), 3, (0, 255, 0), thickness=-1)
        return image


    def draw_detection(self, image, boxes, scores):
        """
        绘制检测结果方法，用于在图像上绘制检测到的物体。

        参数:
           image (numpy.ndarray): 输入图像，为BGR格式的numpy数组。
           boxes (list): 包含检测到的边界框的列表。
           scores (list): 包含检测到的置信度的列表。

        返回值:
           numpy.ndarray: 经过绘制检测结果后的图像。
        """
        for box, score in zip(boxes, scores):
            xmin, ymin, xmax, ymax = box.astype(int)

            # 绘制矩形框
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
            label = "face:" + str(round(score, 2))
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return image

if __name__ == '__main__':
    # 运行shell脚本填写参数
    # python yolov8face.py --imgpath images/4.jpg --confThreshold 0.5
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='4.jpg', help="image path")
    parser.add_argument('--confThreshold', default=0.5, type=float, help='class confidence')
    args = parser.parse_args()

    # 初始化YOLOface_8n对象检测器
    mynet = YOLOface_8n("weights/yoloface_8n.onnx", conf_thres=args.confThreshold)
    srcimg = cv2.imread(args.imgpath)

    # 检测物体
    boxes, kpts, scores = mynet.detect(srcimg)

    # 绘制检测结果
    dstimg = mynet.draw_detections(srcimg, boxes, kpts, scores)
    # dstimg = mynet.draw_detection(srcimg, boxes, scores)
    winName = 'Deep learning yolov8face detection in ONNXRuntime'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
