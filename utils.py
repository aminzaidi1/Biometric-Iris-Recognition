import math
import numpy as np
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt
import cv2
import glob
import os


# Global variable for gamma correction
gamma = 0.32

# Function to convert polar coordinates to Cartesian coordinates
def polar2cart(r, x0, y0, theta):
    x = int(x0 + r * math.cos(theta))
    y = int(y0 + r * math.sin(theta))
    return x, y

# Gamma correction function
def gammaCorrection(image):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255) 
    res = cv2.LUT(image, lookUpTable)
    return res

# Unravel iris function to generate normalized iris
def unravel_iris(img, xp, yp, rp, xi, yi, ri, phase_width=300, iris_width=150):
    if img.ndim > 2:
        img = img[:, :, 0].copy()
    iris = np.zeros((iris_width, phase_width))
    theta = np.linspace(0, 2 * np.pi, phase_width)

    # Generate iris by calculating pixel values for each phase
    for i in range(phase_width):
        begin = polar2cart(rp, xp, yp, theta[i])
        end = polar2cart(ri, xi, yi, theta[i])
        xspace = np.linspace(begin[0], end[0], iris_width)
        yspace = np.linspace(begin[1], end[1], iris_width)
        iris[:, i] = [255 - img[int(y), int(x)] if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0] else 0
                      for x, y in zip(xspace, yspace)]
    return iris

# 2D Gabor wavelets equation
def gabor(rho, phi, w, theta0, r0, alpha, beta):
    return np.exp(-w * 1j * (theta0 - phi)) * np.exp(-(rho - r0) ** 2 / alpha ** 2) * np.exp(-(-phi + theta0) ** 2 / beta ** 2)

# Applying 2D Gabor wavelets on the image
def gabor_convolve(img, w, alpha, beta):
    rho = np.array([np.linspace(0, 1, img.shape[0]) for i in range(img.shape[1])]).T
    x = np.linspace(0, 1, img.shape[0])
    y = np.linspace(-np.pi, np.pi, img.shape[1])
    xx, yy = np.meshgrid(x, y)
    return rho * img * np.real(gabor(xx, yy, w, 0, 0, alpha, beta).T), \
           rho * img * np.imag(gabor(xx, yy, w, 0, 0, alpha, beta).T)

# Iris encoding function
def iris_encode(img, dr=15, dtheta=15, alpha=0.4):
    mask = view_as_blocks(np.logical_and(20 < img, img < 255), (dr, dtheta))
    norm_iris = (img - img.mean()) / img.std()
    patches = view_as_blocks(norm_iris, (dr, dtheta))
    code = np.zeros((patches.shape[0] * 3, patches.shape[1] * 2))
    code_mask = np.zeros((patches.shape[0] * 3, patches.shape[1] * 2))
    
    # Encode iris features using 2D Gabor wavelets
    for i, row in enumerate(patches):
        for j, p in enumerate(row):
            for k, w in enumerate([8, 16, 32]):
                wavelet = gabor_convolve(p, w, alpha, 1 / alpha)
                code[3 * i + k, 2 * j] = np.sum(wavelet[0])
                code[3 * i + k, 2 * j + 1] = np.sum(wavelet[1])
                code_mask[3 * i + k, 2 * j] = code_mask[3 * i + k, 2 * j + 1] = \
                    1 if mask[i, j].sum() > dr * dtheta * 3 / 4 else 0
    code[code >= 0] = 1
    code[code < 0] = 0
    return code, code_mask

# Preprocess function to convert image to grayscale and apply Median Blur filter
def preprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.medianBlur(img, 5)

# Function to find pupil using Hough Circle Transform
def find_pupil_hough(img):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=60, param2=30, minRadius=1, maxRadius=40)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0, 0][0], circles[0, 0][1], circles[0, 0][2]
    else:
        return 0, 0, 0

# Function to find iris using Hough Circle Transform
def find_iris_hough(img):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=60, param2=30, minRadius=20, maxRadius=100)
    if circles is not None: 
        circles = np.uint16(np.around(circles))
        return circles[0, 0][0], circles[0, 0][1], circles[0, 0][2] 
    else:
        return 0, 0, 0

# Function to compare iris codes
def compare_codes(a, b, mask_a, mask_b):
    return np.sum(np.remainder(a + b, 2) * mask_a * mask_b) / np.sum(mask_a * mask_b)

# Function to encode iris features of an image
def encode_photo(image):
    src=gammaCorrection(image)
    newImage = src.copy()
    img = preprocess(image)
    img1 = preprocess(newImage)
    
    x, y, r = find_pupil_hough(img1)
    x_iris, y_iris, r_iris = find_iris_hough(img)
    
    iris = unravel_iris(image, x, y, r, x_iris, y_iris, r_iris)
    return iris_encode(iris)


# Function to select an image from the dataset for matching
def select_image_from_dataset(dataset_folder):
    individual_id = input("Enter individual ID (001-108): ").zfill(3)
    session_id = input("Enter session ID (1-2): ")
    image_id = input("Enter image ID (1-3 for session 1, 1-4 for session 2): ")

    if session_id == '1' and not (1 <= int(image_id) <= 3):
        print("Error: Session 1 only accepts images 1 to 3.")
        return None, None
    elif session_id == '2' and not (1 <= int(image_id) <= 4):
        print("Error: Session 2 only accepts images 1 to 4.")
        return None, None

    image_path = os.path.join(dataset_folder, f"{individual_id}_{session_id}_{image_id}.jpg")
    if not os.path.exists(image_path):
        print("Image not found. Please make sure the individual ID, session ID, and image ID are correct.")
        return None, None
    
    print(f"Selected image: {image_path}")
    image = cv2.imread(image_path)
    return image, individual_id, session_id

# Function to display a list of images
def display_images(image_list):
    for i, img in enumerate(image_list):
        cv2.imshow(f"Image {i+1}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to find the best match in the dataset for a query image
def find_best_match(dataset_folder, query_image):
    best_similarity = 1.0
    best_match_id = None
    
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".jpg"):
            filepath = os.path.join(dataset_folder, filename)
            print(f"Processing image: {filename}")
            dataset_image = cv2.imread(filepath)
            
            # Encode iris features
            query_code, query_mask = encode_photo(query_image)
            dataset_code, dataset_mask = encode_photo(dataset_image)
            
            # Compare iris codes
            similarity = compare_codes(query_code, dataset_code, query_mask, dataset_mask)
            print(f"Similarity with {filename}: {similarity}")
            
            if similarity == 0:
                best_match_id = filename[:3]
                best_similarity = similarity
                break
            
            if similarity < best_similarity:
                best_similarity = similarity
                best_match_id = filename[:3]  # Extract individual ID from filename

    return best_match_id, best_similarity

# Function to display segmentation and normalization process details
def show_details(image):
    src = gammaCorrection(image)
    newImage = src.copy()
    img = preprocess(image)
    img1 = preprocess(newImage)

    x, y, r = find_pupil_hough(img1)
    x_iris, y_iris, r_iris = find_iris_hough(img)

    iris = unravel_iris(image, x, y, r, x_iris, y_iris, r_iris)
    cv2.circle(image, (x, y), r, (255, 0, 0), 3)
    cv2.circle(image, (x, y), 2, (255, 0, 0), 2)
    cv2.circle(image, (x_iris, y_iris), r_iris, (0, 255, 0), 3)
    cv2.circle(image, (x_iris, y_iris), 2, (0, 255, 0), 2)

    f, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(image, cmap=plt.cm.gray)
    axes[0, 0].set_title('Segmentation process')
    axes[0, 1].imshow(iris, cmap=plt.cm.gray)
    axes[0, 1].set_title('Normalization process')

    code, mask = iris_encode(iris)
    axes[1, 0].imshow(iris, cmap=plt.cm.gray)
    axes[1, 0].set_title('Normalization process')
    axes[1, 1].imshow(mask, cmap=plt.cm.gray, interpolation='none')
    axes[1, 1].set_title('Mask code')

    cv2.circle(image, (x, y), r, (255, 255, 0), 2)
    cv2.circle(image, (x_iris, y_iris), r_iris, (0, 255, 0), 2)

    plt.show()