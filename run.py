from tkinter import Tk, Button, Label, Entry, Canvas, messagebox
from tkinter.filedialog import askopenfilename

import PIL
from PIL import ImageTk
import cv2
import matplotlib.pyplot as plt  ###如无则pip安装
from yolov8face import YOLOface_8n
from face_68landmarks import face_68_landmarks
from face_recognizer import face_recognize
from face_swap import swap_face
from face_enhancer import enhance_face

# 创建主窗口
root = Tk()
root.title('人脸表情合成')
root.geometry('1000x520')


global path1_, path2_, seg_img_path


path1 = ""  # 第一张脸部图片的路径
path2 = ""  # 第二张脸部图片的路径
output_path = 'output/output.jpg'  # 输出图片的路径


# 测试函数，合成两张图片并保存输出图片
def main():
    source_path = path1_
    target_path = path2_
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)

    detect_face_net = YOLOface_8n("weights/yoloface_8n.onnx")
    detect_68landmarks_net = face_68_landmarks("weights/2dfan4.onnx")
    face_embedding_net = face_recognize('weights/arcface_w600k_r50.onnx')
    swap_face_net = swap_face('weights/inswapper_128.onnx')
    enhance_face_net = enhance_face('weights/gfpgan_1.4.onnx')

    boxes, _, _ = detect_face_net.detect(source_img)
    position = 0  ###一张图片里可能有多个人脸，这里只考虑1个人脸的情况
    bounding_box = boxes[position]
    _, face_landmark_5of68 = detect_68landmarks_net.detect(source_img, bounding_box)
    source_face_embedding, _ = face_embedding_net.detect(source_img, face_landmark_5of68)

    boxes, _, _ = detect_face_net.detect(target_img)
    position = 0  ###一张图片里可能有多个人脸，这里只考虑1个人脸的情况
    bounding_box = boxes[position]
    _, target_landmark_5 = detect_68landmarks_net.detect(target_img, bounding_box)

    swapimg = swap_face_net.process(target_img, source_face_embedding, target_landmark_5)
    resultimg = enhance_face_net.process(swapimg, target_landmark_5)

    plt.subplot(1, 2, 1)
    plt.imshow(source_img[:, :, ::-1])  ###plt库显示图像是RGB顺序
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(target_img[:, :, ::-1])
    plt.axis('off')
    # plt.show()
    plt.savefig('source_target.jpg', dpi=600, bbox_inches='tight')  ###保存高清图

    # 将合成后的图片显示在界面上
    cv2.imwrite(output_path, resultimg)
    # 输出文件的路径存储
    mor_img_path = output_path
    # 打开图像文件
    Img = PIL.Image.open(r'{}'.format(mor_img_path))
    # 调整图片大小
    Img = Img.resize((270, 270), PIL.Image.LANCZOS)
    # 使用 ImageTk.PhotoImage 将 PIL 图像转换为 Tkinter 图像。
    img_png_seg = ImageTk.PhotoImage(Img)
    # 将 Tkinter 图像配置到界面上的 label_Img_seg 控件中，以在界面上显示图像。
    label_Img_seg.config(image=img_png_seg)
    label_Img_seg.image = img_png_seg


# 打开图片1的回调函数
def openFirstPhoto():
    global path1_
    # 弹出文件选择对话框获取文件路径
    path = askopenfilename(title='选择文件')
    if (path is None) or (path == ''):
        return
    path1_ = path
    # 打印图片1路径
    print(path1_)
    # 使用Python Imaging Library（PIL）中的Image.open()函数来打开指定路径的图像文件。
    # r'{}'.format(path1_)是一个字符串格式化的语法，用来将变量path1_的值插入到字符串中
    try:
        Img = PIL.Image.open(r'{}'.format(path1_))
    except:
        return

    # 调整图片大小至256x256
    # 使用Lanczos滤波器进行图像重采样，可以在调整大小时保持图像质量。
    Img = Img.resize((270, 270), PIL.Image.LANCZOS)
    # 使用ImageTk模块中的PhotoImage类，将PIL库中的图像对象Img转换为Tkinter的图像对象img_png_original。
    # 在Tkinter中显示图像。
    img_png_original = ImageTk.PhotoImage(Img)
    # 将Tkinter的标签组件label_Img_original1的图像配置为之前创建的img_png_original图像对象
    label_Img_original1.config(image=img_png_original)
    # 在Tkinter窗口中显示图像。
    label_Img_original1.image = img_png_original
    # 在Canvas上创建一个图像，坐标为(5, 5)，并将其锚定在左上角，然后使用img_png_original作为图像。
    cv_orinial1.create_image(5, 5, anchor='nw', image=img_png_original)
# 打开图片2的回调函数
def openSecondPhoto():
    global path2_
    # 弹出文件选择对话框获取文件路径
    path = askopenfilename(title='选择文件')
    if (path is None) or (path == ''):
        return
    path2_ = path

    # 打印图片2路径
    print(path2_)
    try:
        Img = PIL.Image.open(r'{}'.format(path2_))
    except:
        return
    # 调整图片大小至256x256
    Img = Img.resize((270, 270), PIL.Image.LANCZOS)
    # 在Tkinter中显示图像。
    img_png_original = ImageTk.PhotoImage(Img)
    label_Img_original2.config(image=img_png_original)
    # 在Tkinter窗口中显示图像。
    label_Img_original2.image = img_png_original
    cv_orinial2.create_image(5, 5, anchor='nw', image=img_png_original)
# 置换按钮的回调函数
def photoReplacement():
    global path1_,path2_
    try:
        # 交换两张照片路径
        temp = path1_
        path1_ = path2_
        path2_ = temp
    except:
        return
    img1 = PIL.Image.open(r'{}'.format(path1_))
    # 调整图片大小至270
    img1 = img1.resize((270, 270), PIL.Image.LANCZOS)
    # 在Tkinter中显示图像。
    img1Original = ImageTk.PhotoImage(img1)
    label_Img_original1.config(image=img1Original)
    # 在Tkinter窗口中显示图像。
    label_Img_original1.image = img1Original
    cv_orinial1.create_image(5, 5, anchor='nw', image=img1Original)

    img2 = PIL.Image.open(r'{}'.format(path2_))
    # 调整图片大小至270
    img2 = img2.resize((270, 270), PIL.Image.LANCZOS)
    # 在Tkinter中显示图像。
    img2Original = ImageTk.PhotoImage(img2)
    label_Img_original2.config(image=img2Original)
    # 在Tkinter窗口中显示图像。
    label_Img_original2.image = img2Original
    cv_orinial2.create_image(5, 5, anchor='nw', image=img2Original)
# 退出软件的回调函数
def quit():
    root.destroy()

# 创建按钮和标签
Button(root, text="打开图片1", command=openFirstPhoto).place(x=150, y=440)
Button(root, text="打开图片2", command=openSecondPhoto).place(x=480, y=440)
Button(root, text="人脸融合", command=main).place(x=800, y=440)
Button(root, text="退出软件", command=quit).place(x=800, y=40)
Button(root,text="⇔", command=photoReplacement).place(x=340,y=260)

# 图片1展示
Label(root, text="图片1", font=10).place(x=150, y=120)
cv_orinial1 = Canvas(root, bg='white', width=270, height=270)
cv_orinial1.create_rectangle(8, 8, 260, 260, width=1, outline='black')
cv_orinial1.place(x=50, y=150)
label_Img_original1 = Label(root)
label_Img_original1.place(x=50, y=150)
# 图片2展示
Label(root, text="图片2", font=10).place(x=480, y=120)
cv_orinial2 = Canvas(root, bg='white', width=270, height=270)
cv_orinial2.create_rectangle(8, 8, 260, 260, width=1, outline='black')
cv_orinial2.place(x=380, y=150)
label_Img_original2 = Label(root)
label_Img_original2.place(x=380, y=150)
# 融合效果展示
Label(root, text="融合效果", font=10).place(x=800, y=120)
cv_seg = Canvas(root, bg='white', width=270, height=270)
cv_seg.create_rectangle(8, 8, 260, 260, width=1, outline='black')
cv_seg.place(x=700, y=150)
label_Img_seg = Label(root)
label_Img_seg.place(x=700, y=150)

# 进入主循环
root.mainloop()
