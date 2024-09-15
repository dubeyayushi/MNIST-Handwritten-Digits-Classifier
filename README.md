# MNIST Handwritten Digits Classifier

## Overview

This project is a handwritten digits classifier built using PyTorch. It leverages the MNIST dataset and demonstrates a machine learning pipeline from data loading and preprocessing to model training and evaluation. The project also includes a user-friendly web interface built with Streamlit, allowing users to interactively test the model with their own handwritten digit inputs.

## Model

The classifier is built using a Convolutional Neural Network (CNN), a common architecture for image classification tasks. The model is designed to recognize digits from 0 to 9 based on the MNIST dataset.

### Key Components:

- **Architecture:** The CNN model consists of several convolutional layers followed by fully connected layers.
- **Framework:** PyTorch is used for model implementation and training.

## Dataset

The model is trained and evaluated on the MNIST dataset, which consists of:

- **Training Set:** 60,000 images of handwritten digits.
- **Test Set:** 10,000 images of handwritten digits.

The images are grayscale and of size 28x28 pixels.

## Results

The model achieves high accuracy on the MNIST test set. Performance metrics and accuracy details are logged during training and evaluation phases.

## Streamlit App 

You can access the Streamlit app here: [dubeyayushi-mnist-handwritten-digits-classifie-mnist-app-plssff.streamlit.app](https://dubeyayushi-mnist-handwritten-digits-classifie-mnist-app-plssff.streamlit.app/)

The Streamlit application provides an interactive interface for users to test the classifier. Users can:

- **Upload an Image:** Load a handwritten digit image.
- **Predict Digit:** The model will predict the digit and display the result.

### Features:

- **Image Upload:** Drag and drop or browse to select an image file.
- **Prediction:** Real-time classification and display of results.

## Setup

### Prerequisites

Ensure you have Python 3.8 or higher installed.

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/handwritten-digits-classifier.git
   cd handwritten-digits-classifier
  
2. **Create a Virtual Environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install Dependencies:**

   Make sure your requirements.txt does not specify strict versions if you want pip      to handle compatibility automatically.

   ```bash
   pip install -r requirements.txt

4. **Run the Streamlit App:**

   ```bash
   streamlit run mnist_app.py

## Usage

1. **Open the Streamlit App:**
   Follow the instructions above to start the app.

2. **Upload a Handwritten Digit Image:**
   Use the upload button to select an image file from your computer.

3. **View Prediction:**
   The model will predict the digit and display the result on the screen.

## Contributing

Contributions are welcome! If you have suggestions, improvements, or fixes, please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MNIST Dataset:** A classic dataset used for training image processing systems.
- **PyTorch:** For providing the deep learning framework.
- **Streamlit:** For enabling easy web app creation and deployment.

This project was completed as part of the advanced Machine Learning Nanodegree Program by Udacity and AWS.



  
