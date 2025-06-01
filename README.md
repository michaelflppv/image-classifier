# Image Classification Project: Real vs Fake Image Detector

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images as either "REAL" or "FAKE". The model uses multiple convolutional layers with data augmentation techniques to improve classification accuracy.

## Project Structure

- `src/main.py`: Main Python script containing model architecture, training and testing functions
- `archive/train/`: Directory containing training images organized in class subdirectories
- `trained_model.keras`: Saved model file after training (stored using Git LFS)

## Features

- Binary image classification (REAL/FAKE)
- Data augmentation (rotation, shifting, zoom, flip) to improve model generalization
- Early stopping to prevent overfitting
- Interactive CLI for training and testing
- Confidence score reporting for predictions

## Requirements

```
tensorflow==2.19.0
numpy==2.1.3
keras==3.10.0
pillow==11.2.1
scipy==1.11.4
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/image-classification.git
   cd image-classification
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Pull the large files (if using the pre-trained model):
   ```bash
   git lfs pull
   ```

## Usage

Run the main script:
```bash
python src/main.py
```

### Training Mode

Select "train" when prompted to train the model:
- Images should be in the `archive/train` directory with appropriate class subdirectories
- Default hyperparameters: batch_size=32, epochs=10
- The model architecture uses three convolutional layers, followed by dense layers
- Training progress and validation accuracy will be displayed
- The trained model will be saved as `trained_model.keras`

### Testing Mode

Select "test" when prompted to test the model:
- You will be asked to provide the path to an image for classification
- The model will process the image and provide:
  - Classification result (REAL or FAKE)
  - Confidence percentage
  - Raw prediction value

## Git LFS Usage

This repository uses Git Large File Storage (Git LFS) for managing large files such as:
- The trained model file (`trained_model.keras`)
- Any large dataset files

To work with this repository, ensure you have Git LFS installed:
```bash
git lfs install
```

## Model Architecture

The CNN architecture consists of:
- 3 convolutional layers with ReLU activation and max pooling
- Flatten layer to convert 2D feature maps to 1D features
- 2 dense layers with the final layer using sigmoid activation for binary classification
- Adam optimizer with binary crossentropy loss