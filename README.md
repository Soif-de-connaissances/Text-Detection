# Text-Detection

## Project Goal
The goal of this project is to use PyCharm to detect text in images using OpenCV. To achieve this,  utilize the OpenCV package and the pyplot package from Matplotlib. Additionally, I selected two OCR packages to meet the requirements for text recognition and extraction. 

## Environment Setup
### General Environment Setup
The main modules required are OpenCV and Matplotlib. You can install them using one of the following methods:

**Method 1**: Add them in PyCharm:
- Go to `Files → Settings → Project → Python Interpreter`.

**Method 2**: Use the pip install command in the command prompt:
```bash
pip install opencv-python matplotlib
```

### OCR Features Overview
#### 1. Tesseract OCR Features:
- **Deep Learning Support**: Tesseract OCR introduced LSTM (Long Short-Term Memory networks) as a deep learning model starting from version 4.x to improve recognition accuracy.
- **Model Types**: Tesseract supports multiple languages and can be extended to support specific fonts or languages by training new datasets.
- **Configuration and Usage**: Although Tesseract supports deep learning, its configuration and usage are relatively complex, requiring users to understand how to train and optimize the model.

#### 2. PaddleOCR Features:
- **Deep Learning Support**: PaddleOCR is built on the PaddlePaddle deep learning framework and utilizes various deep learning models for text detection and recognition.
- **Model Types**: PaddleOCR offers multiple pre-trained models that support various languages and scenarios, including Chinese and English. It employs advanced architectures such as CRNN (Convolutional Recurrent Neural Network) to enhance recognition accuracy.
- **Flexibility**: Users can choose different models and parameters based on their needs and can even customize and train their own models.

## Usage Instructions
### Tesseract OCR
First, you need to download Tesseract OCR. There are two methods:
**Method 1**: Download from GitHub:
[Download Link](https://github.com/tesseract-ocr/tesseract)

**Method 2**: Execute the command in the command prompt:
```bash
pip install pytesseract
```

#### Example Code
```python
import cv2
import pytesseract
import numpy as np
from pytesseract import Output
from matplotlib import pyplot as plt

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

plt.rcParams['figure.figsize'] = (10.0, 8.0)
raw_img = cv2.imread('Test_image1_en.PNG')

custom_config = r'--oem 3 --psm 3 -l eng+chi_sim'
text = pytesseract.image_to_string(raw_img, config=custom_config)

gray_img = get_grayscale(raw_img)
thresh_img = thresholding(gray_img)
data = pytesseract.image_to_data(thresh_img, output_type=Output.DICT)

# Draw rectangles around recognized text
n_boxes = len(data['text'])
for i in range(n_boxes):
    if int(data['conf'][i]) > 60:
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        img = cv2.rectangle(raw_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.imshow(img)
plt.axis('off')
plt.show()
```

### PaddleOCR
First, you need to download PaddleOCR. There are two methods:
**Method 1**: Download from GitHub:
[Download Link](https://github.com/PaddlePaddle/PaddleOCR)

**Method 2**: Execute the command in the command prompt:
```bash
pip install paddlepaddle paddleocr
```

#### Example Code
```python
from paddleocr import PaddleOCR, draw_ocr
import cv2
import matplotlib.pyplot as plt

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.rcParams['figure.figsize'] = (10.0, 8.0)
raw_img = cv2.imread('Test_image1_en.PNG')

ocr = PaddleOCR(use_angle_cls=True, lang='ch')
result = ocr.ocr(raw_img, cls=True)

# Process results and draw
for line in result[0]:
    print(f"Text: {line[1][0]}, Confidence: {line[1][1]}")
```

## Efficiency and Effectiveness Evaluation
### Tesseract OCR
- **Efficiency**: The system processes images relatively quickly, completing text detection tasks in less than 10 seconds.
- **Effectiveness**: By calculating the average confidence level (0.9533), the system demonstrates high accuracy in text recognition.

### PaddleOCR
- **Efficiency**: The system processes images quickly, completing text detection tasks in less than 10 seconds.
- **Effectiveness**: By calculating the average confidence level (0.9940), the system demonstrates even higher accuracy in text recognition.

## Summary
Both OCRs exhibit high text recognition performance; however, PaddleOCR comes with multiple pre-trained models, while Tesseract's configuration and usage are relatively complex, requiring users to understand how to train and optimize the model. Therefore, for beginners, I recommend using PaddleOCR, while for professionals, I believe both are equally effective.

## Potential Improvement Suggestions
Although Tesseract OCR and PaddleOCR perform well in text detection, there is still room for improvement:
- **Enhance Preprocessing Steps**: Different image processing techniques, such as adaptive thresholding and additional morphological operations, can be tried to improve text visibility.
- **Model Training**: Training new datasets can enhance support for specific fonts or languages, thereby improving recognition accuracy.

---
