# 在GitHub中安装PaddleOCR模型，或者在cmd中输入pip install paddlepaddle paddleocr
# PaddleOCR
# 深度学习支持：PaddleOCR 是基于 PaddlePaddle 深度学习框架构建的，使用了多种深度学习模型进行文本检测和识别。
# 模型类型：PaddleOCR 提供了多种预训练的模型，支持多种语言和场景，包括中文、英文等。它使用了如 CRNN（卷积递归神经网络）和其他先进的深度学习架构来提高识别精度。
# 灵活性：用户可以根据需求选择不同的模型和参数，甚至可以自定义训练自己的模型。
from paddleocr import PaddleOCR, draw_ocr
import cv2
import matplotlib.pyplot as plt

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)

# 控制图形的外形大小
plt.rcParams['figure.figsize'] = (10.0, 8.0)

# 读取图像并保存为 raw_image
raw_img = cv2.imread('Test_image1_en.PNG')

# 检查读取图像是否成功，成功则显示
if raw_img is None:
    print("Error: Unable to load image.")
else:
    # cv2默认将读取的图像以BGR的颜色通道顺序保存，这里将其转换为RGB
    plt.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')  # 关闭坐标轴
    # 在Pycharm环境中需要借助show函数来显示图像
    plt.show()

# 对图像进行高斯模糊处理
d = 3
processed_img = raw_img.copy()
processed_img = cv2.GaussianBlur(processed_img, (2*d+1, 2*d+1), -1)[d:-d, d:-d]

blur_img = processed_img.copy()

# Canny边缘检测
# 先将模糊化的图像转为灰度图方便Canny边缘检测
gray_image = get_grayscale(processed_img)
th1 = 30
th2 = 90
edges = cv2.Canny(gray_image, th1, th2)
plt.imshow(edges, cmap='gray')
plt.title('Edges Image')
plt.axis('off')
plt.show()

# 将边缘检测的结果图显示出来
processed_img[edges != 0] = (0, 255, 0)
plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
plt.title('Processed Edges Detection Image')
plt.axis('off')
plt.show()

# 初始化PaddleOCR
# PaddleOCR 有一个缺点就是无法同时使用两种语言模型，
ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 使用角度分类器和英语模型

# 读取blur图像
new_img = blur_img.copy()
img_rgb = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

# 执行OCR
result = ocr.ocr(img_rgb, cls=True)  # 文本方向分类器

# 打印原始结果以进行调试
print("Original OCR result:")
print(result)

# 检查结果是否为空
if not result or len(result) == 0:
    print("No text detected in the image.")
else:
    # 初始化列表
    boxes = []  # 文本框坐标
    txts = []  # 文本内容
    scores = []  # 置信度

    # 存储OCR结果
    for line in result[0]:  # result[0]代表第一张图OCR的结果，如果OCR了两张图，result[1]代表第二张图的OCR结果
        boxes.append(line[0])  # line[0]固定存储文本框坐标
        txts.append(line[1][0])  # line[1][0]固定存储文本内容
        scores.append(line[1][1])  # line[1][1]固定存储OCR置信度

    # 打印处理后的结果
    print(f"Processed {len(boxes)} text regions")

    # 检查boxes是否valid并绘制结果
    if boxes:
        show_img = draw_ocr(img_rgb, boxes, txts, scores)  # draw_ocr 主要用于可视化 OCR 识别的结果

        # 显示分析结果图片
        plt.imshow(show_img)
        plt.title('OCR Result')
        plt.axis('off')
        plt.show()

        # 打印识别的文本以及它的位置和置信度
        # zip 函数用于将多个可迭代对象（如列表）组合在一起，形成一个元组的迭代器
        # 如果 boxes = [[1, 2], [3, 4]]，txts = ["文本1", "文本2"]，scores = [0.9, 0.85]，
        # 那么 zip(boxes, txts, scores) 将生成 [((1, 2), "文本1", 0.9), ((3, 4), "文本2", 0.85)]
        # enumerate 函数用于将一个可迭代对象转换为一个索引序列。在遍历时，它会返回每个元素的索引和元素本身
        # 如果 zip 生成的结果有两个元组，enumerate 将返回 (0, 元组1) 和 (1, 元组2)
        for idx, (box, txt, score) in enumerate(zip(boxes, txts, scores)):
            print(f"文本 {idx+1}:")
            print(f"位置: {box}")
            print(f"识别的文本: {txt}")
            print(f"置信度: {score:.4f}")
            print("--------------------")

        # 连续打印文本
        print("文本整体打印：")
        for txt in txts:
            print(txt, end=' ')

        # 计算平均置信度
        if scores:
            average_confidence = sum(scores) / len(scores)
            print(f"\nAverage Confidence: {average_confidence:.4f}")
        else:
            print("No valid scores to calculate average confidence.")

    else:
        print("No valid text boxes found in the result.")