# Iris Recognition Project

This project implements an iris recognition system using Python and OpenCV. It provides functionality to preprocess iris images, extract iris features, compare iris codes, and find the best match in a dataset.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Credits](#credits)

## Introduction

Iris recognition stands as a premier biometric authentication method, leveraging the distinctive patterns within the human iris for identification. This innovative project endeavors to craft a robust iris recognition system primed for diverse applications, including access control, security systems, and personal identification. Harnessing advanced image processing techniques and machine learning algorithms, the system promises heightened security and seamless user authentication, positioning it as a cornerstone in the realm of biometric authentication technologies.

## Features

- Preprocessing of iris images
- Iris feature extraction using 2D Gabor wavelets
- Encoding of iris features into binary codes
- Comparison of iris codes for matching
- Visualization of segmentation and normalization process details
- Interactive selection of images from a dataset for matching

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/aminzaidi1/iris-recognition.git

2. Navigate to the project directory:

   ```python
   cd iris-recognition

3. Install the required dependencies:

   ```python
   pip install -r requirements.txt

## Usage

1. Run the main.py script to execute the iris recognition system:

    ```python
    python main.py

2. Follow the on-screen instructions to select an image from the dataset for matching.

## Dataset

This project utilizes the CASIA-IrisV1 dataset, a widely used dataset for iris recognition research. The CASIA-IrisV1 dataset provides a comprehensive collection of iris images. It consists of 7 iris images of 108 individuals from 2 sessions. 3 images from the 1st session and 4 from the 2nd session.

**Download the CASIA-IrisV1 dataset**: [CASIA-IrisV1](http://biometrics.idealtest.org/downloadDB.do?id=4&subset=1#/)

Ensure that you have the necessary permissions to use the dataset in accordance with its licensing terms.

### Dataset Organization

The dataset folder contains images of iris samples used for testing the iris recognition system. To ensure smooth functioning of the system, it's crucial to organize the dataset properly and adhere to a specific naming convention for image filenames.

1. **Directory Structure**: Ensure that all iris images are stored within the `dataset` folder.
2. **Naming Convention**: Use the following naming convention for image filenames: `XXX_S_Y`, where:
   - `XXX` represents the individual ID (e.g., 007).
   - `S` represents the session ID (e.g., 1 or 2).
   - `Y` represents the image ID (e.g., 1, 2, 3, etc.).

Example: For an individual with ID 007, session 1, and image 3, the filename should be `007_1_3`.

## Credits

This project was developed by Amin Zaidi for the Capstone Project of my Course of Data Science & Artificial Intelligence by Boston Institute of Analytics (BIA).