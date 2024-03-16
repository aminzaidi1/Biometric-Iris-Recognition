# Iris Recognition Project

This project implements an iris recognition system using Python and OpenCV. It provides functionality to preprocess iris images, extract iris features, compare iris codes, and find the best match in a dataset.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Credits](#credits)
- [License](#license)

## Introduction

Iris recognition is a biometric authentication method that uses the unique patterns in the iris of the human eye for identification. This project aims to develop an iris recognition system that can be used for various applications such as access control, security systems, and personal identification.

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
   git clone https://github.com/yourusername/iris-recognition.git

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

The dataset folder contains images of iris samples used for testing the iris recognition system. Make sure to organize the dataset properly and ensure that image filenames follow the specified naming convention XXX_Y_S. XXX being individual_id, Y being session_id, and S being image_id. For example: 007_1_3.

## Credits

This project was developed by Amin Zaidi for the Capstone Project of my Course of Data Science & Artificial Intelligence by Boston Institute of Analytics(BIA).