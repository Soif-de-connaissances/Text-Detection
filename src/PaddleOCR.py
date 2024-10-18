# PaddleOCR Characteristics:
# Deep Learning Support: PaddleOCR is built on the PaddlePaddle deep learning framework and utilizes various deep learning models for text detection and recognition.
# Model Types: PaddleOCR offers multiple pre-trained models that support various languages and scenarios, including Chinese and English. It employs advanced deep learning architectures such as CRNN (Convolutional Recurrent Neural Network) to enhance recognition accuracy.
# Flexibility: Users can choose different models and parameters based on their needs and can even customize and train their own models.
from paddleocr import PaddleOCR, draw_ocr
import cv2
import matplotlib.pyplot as plt

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Control the size of the figure
plt.rcParams['figure.figsize'] = (10.0, 8.0)

# Read the image and save it as raw_image
raw_img = cv2.imread('Test_image1_en.PNG')

# Check if the image is loaded successfully, if so, display it
if raw_img is None:
    print("Error: Unable to load image.")
else:
    # OpenCV reads the image in BGR color channel order by default, here we convert it to RGB
    plt.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')  # Turn off the axis
    # In the PyCharm environment, the show function is needed to display the image
    plt.show()

# Apply Gaussian blur to the image
d = 3
processed_img = raw_img.copy()
processed_img = cv2.GaussianBlur(processed_img, (2*d+1, 2*d+1), -1)[d:-d, d:-d]

blur_img = processed_img.copy()

# Canny边缘检测
# Convert the blurred image to grayscale for easier Canny edge detection
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

# Initialize PaddleOCR
# One downside of PaddleOCR is that it cannot use two language models simultaneously
ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # Use angle classifier and English model

# Read the blurred image
new_img = blur_img.copy()
img_rgb = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

# Perform OCR
result = ocr.ocr(img_rgb, cls=True)  # cls:Text direction classifier

# Print the original result for debugging
print("Original OCR result:")
print(result)

# Check if the result is empty
if not result or len(result) == 0:
    print("No text detected in the image.")
else:
    # Initialize lists
    boxes = []  # Text box coordinates
    txts = []  # Text content
    scores = []  # Confidence scores

    # Store the OCR results
    # result[0] represents the OCR result for the first image;
    # if two images were OCR'd, result[1] represents the second image's result
    for line in result[0]:
        boxes.append(line[0])  # line[0] stores text box coordinates
        txts.append(line[1][0])  # line[1][0] stores text content
        scores.append(line[1][1])  # line[1][1] stores OCR confidence score

    # Print the processed results
    print(f"Processed {len(boxes)} text regions")

    # Check if boxes are valid and draw the results
    if boxes:
        # draw_ocr is mainly used to visualize the OCR recognition results
        show_img = draw_ocr(img_rgb, boxes, txts, scores)

        # Display the analysis result image
        plt.imshow(show_img)
        plt.title('OCR Result')
        plt.axis('off')
        plt.show()

        # Print the recognized text along with its position and confidence
        # The zip function is used to combine multiple iterables (like lists) into an iterator of tuples
        # If boxes = [[1, 2], [3, 4]], txts = ["Text1", "Text2"], scores = [0.9, 0.85],
        # then zip(boxes, txts, scores) will generate [((1, 2), "Text1", 0.9), ((3, 4), "Text2", 0.85)]
        # The enumerate function is used to convert an iterable into an index sequence.
        # During iteration, it returns the index and the element itself
        # If zip generates two tuples, enumerate will return (0, tuple1) and (1, tuple2)
        for idx, (box, txt, score) in enumerate(zip(boxes, txts, scores)):
            print(f"Text {idx+1}:")
            print(f"Position: {box}")
            print(f"Recognized Text: {txt}")
            print(f"Confidence: {score:.4f}")
            print("--------------------")

        # Print all recognized text continuously
        print("Confidence：")
        for txt in txts:
            print(txt, end=' ')

        # Calculate average confidence
        if scores:
            average_confidence = sum(scores) / len(scores)
            print(f"\nAverage Confidence: {average_confidence:.4f}")
        else:
            print("No valid scores to calculate average confidence.")

    else:
        print("No valid text boxes found in the result.")