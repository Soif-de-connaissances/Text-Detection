# Tesseract OCR Characteristics:
# Deep Learning Support: Tesseract OCR previously relied mainly on traditional image processing techniques for OCR. However, starting from version 4.x, it introduced LSTM (Long Short-Term Memory networks) as a deep learning model to improve recognition accuracy.
# Model Types: Tesseract supports multiple languages and can be extended to support specific fonts or languages by training new datasets.
# Configuration and Usage: Although Tesseract supports deep learning, its configuration and usage are relatively complex, requiring users to understand how to train and optimize the model.

import cv2
import pytesseract
from pytesseract import Output
from matplotlib import pyplot as plt

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Set the figure size for plotting
# This piece of code sets the height of the image to 1000 pixels and the width to 800 pixels.
plt.rcParams['figure.figsize'] = (10.0, 8.0)

raw_img = cv2.imread('Test_image1_en.PNG')

# Adding custom options
# oem: Specifies the OCR Engine Mode.
# The specific modes of OEM (OCR Engine Mode) are as follows:
# 0: Use the traditional Tesseract OCR engine.
# 1: Use only LSTM (Long Short-Term Memory).
# 2: Use a combination of traditional engine and LSTM engine.
# 3: Support both LSTM neural network and traditional OCR engine.

# psm: Specifies the Page Segmentation Mode.
# psm 3: Automatically detect the number of text blocks and process them. This is a general option suitable for most scenarios.
# psm 4: Assume there are multiple text blocks in the image but do not consider the direction of the lines.
# psm 5: Assume there are multiple text blocks and these text blocks are arranged vertically.
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
# image_to_data method: Returns detailed information about each recognized text box, including top-left coordinates, width, height, and confidence.


# Use image_to_data function to perform OCR on the thresholded image and extract text information
data = pytesseract.image_to_data(thresh_img, output_type=Output.DICT)
# Print the keys of the data dictionary to see the structure of the OCR output
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

# Print all detected text
for j in range(n_boxes):
    print(data['text'][j], end=' ')

# Calculate and print the average confidence
if count > 0:
    average_conf = total_conf / count
    print(f'\nAverage Confidence: {average_conf}')
else:
    print('\nNo valid confidence scores to calculate average.')

plt.imshow(img)
plt.axis('off')
plt.show()