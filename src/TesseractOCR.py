# TesseractOCR
# 深度学习支持：TesseractOCR 以前主要依靠传统的图像处理技术进行 OCR，但从 Tesseract 4.x 版本开始，引入了 LSTM（长短期记忆网络）作为深度学习模型，以提高识别精度。
# 模型类型：Tesseract 提供了多种语言的支持，并且可以通过训练新的数据集来扩展对特定字体或语言的支持。
# 配置和使用：虽然 Tesseract 支持深度学习，但其配置和使用相对较为复杂，用户需要了解如何训练和优化模型。
import cv2
import pytesseract
import numpy as np
from pytesseract import Output
from matplotlib import pyplot as plt

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

# 控制图形的外形大小
# 这段代码相当于将图形的高度设置为了1000bit，宽度设置为了800bit
plt.rcParams['figure.figsize'] = (10.0, 8.0)

raw_img = cv2.imread('Test_image1_en.PNG')

# Adding custom options
# oem：指定 OCR 引擎模式（OCR Engine Mode）。
# oem 的具体模式如下：
# 0：使用传统的 Tesseract OCR 引擎。
# 1：仅使用 LSTM （长短期记忆）。
# 2：使用传统引擎和 LSTM 引擎的组合。
# 3：可以支持 LSTM神经网络和传统的 OCR 引擎

# psm：指定页面分割模式（Page Segmentation Mode）。
# psm 3：自动检测文本块的数量并进行处理。这是一个比较通用的选项，适用于大多数场景。
# psm 4：假设图像中有多个文本块，但不考虑行的方向。
# psm 5：假设图像中有多个文本块，并且这些文本块是垂直排列的
custom_config = r'--oem 3 --psm 3 -l eng+chi_sim'
pytesseract.image_to_string(raw_img, config=custom_config)

gray_img = get_grayscale(raw_img)
plt.imshow(gray_img, cmap='gray')
plt.axis('off')
plt.show()

thresh_img = thresholding(gray_img)
plt.imshow(thresh_img, cmap='gray')
plt.axis('off')
plt.show()

# Getting boxes around text words (image_to_data)
# image_to_data 方法：返回每个识别到的文本框的详细信息，包括左上角坐标、宽度、高度和置信度。


# 使用 image_to_data 函数对读取的图像进行 OCR 处理，提取文字信息
data = pytesseract.image_to_data(thresh_img, output_type=Output.DICT)
# 打印字典 data 的所有键，以查看 OCR 处理后返回的数据结构的组成部分
print(data.keys())

n_boxes = len(data['text'])
total_conf = 0
count = 0
for i in range(n_boxes):
    text = data['text'][i]
    confidence = data['conf'][i]
    if int(data['conf'][i]) > 60:
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        img = cv2.rectangle(raw_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(f'Text: {text}, Confidence: {confidence}')  # 打印识别文本及其置信度
        total_conf += confidence
        count += 1

# 打印所有文本
for j in range(n_boxes):
    print(data['text'][j], end=' ')

# 计算并打印平均置信度
if count > 0:
    average_conf = total_conf / count
    print(f'\nAverage Confidence: {average_conf}')
else:
    print('\nNo valid confidence scores to calculate average.')

plt.imshow(img)
plt.axis('off')
plt.show()