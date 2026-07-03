# Smoker Detector

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

A deep learning project for detecting whether a person is **smoking** or **not smoking** from an image.

The model is built as a Convolutional Neural Network using Keras and trained on a binary image dataset with two classes:

- `smoking`
- `notsmoking`

## Overview

This project demonstrates a full beginner-friendly computer vision workflow:

1. Load image data from folders.
2. Convert images to grayscale.
3. Resize images to `100x100`.
4. Normalize pixel values.
5. Train a CNN classifier.
6. Predict the class of a new image.

## Dataset Structure

The notebook expects the dataset to be organized like this:

```text
images/
|-- Training/
|   `-- Training/
|       |-- smoking_0001.jpg
|       |-- notsmoking_0001.jpg
|       `-- ...
|-- Validation/
|   `-- Validation/
|       |-- smoking_0001.jpg
|       |-- notsmoking_0001.jpg
|       `-- ...
`-- Testing/
    `-- Testing/
        |-- smoking_0001.jpg
        |-- notsmoking_0001.jpg
        `-- ...
```

Labels are detected from the filename prefix:

| Filename prefix | Class |
| --- | --- |
| `smoking_` | Smoking |
| `notsmoking_` | Not smoking |

## Model Architecture

The classifier is a compact CNN with:

- `Conv2D` layers for feature extraction
- `MaxPooling2D` layers for downsampling
- `Flatten` layer before classification
- fully connected `Dense` layers
- final `softmax` layer for binary classification

Input shape:

```text
100 x 100 x 1
```

Output classes:

```text
[notsmoking, smoking]
```

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

## Installation

Clone the repository:

```bash
git clone https://github.com/armanvardanyan07/smoker-detector.git
cd smoker-detector
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open the notebook file and run all cells from top to bottom.

The notebook will:

- load and preprocess training images
- build the CNN model
- train the model
- evaluate performance
- run prediction on a sample image

## Example Prediction Flow

```python
test_image = cv2.resize(test_image, (100, 100))
test_image = test_image / 255
test_image = test_image.reshape(1, 100, 100, 1)

prediction = model.predict(test_image)
```

## Project Status

This repository is intended as a learning project for image classification with deep learning.

Possible improvements:

- add a cleaner training script
- save and load trained model weights
- add accuracy/loss plots
- add a confusion matrix
- add data augmentation
- build a small web demo with Streamlit or Flask

## License

This project is open for educational use.

