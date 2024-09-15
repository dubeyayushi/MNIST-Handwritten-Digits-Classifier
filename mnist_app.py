import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st

# Define the model architecture (same as the one you used for training)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Function to load the model
def load_model():
    model = Net()
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Initialize the Streamlit app
st.title("🖋️ MNIST Handwritten Digit Classifier")

# Create a sidebar with sections and emojis
st.sidebar.title("🔍 App Navigation")
st.sidebar.markdown("Use the sidebar to navigate through different sections of the app.")

section = st.sidebar.radio("📄 Go to:", ["📑 Project Description", "🚀 How to Use the App", "🖼️ Upload Image"])

# Project description with model details
if section == "📑 Project Description":
    st.write("""
    ## 📑 Project Description

    Welcome to the **MNIST Handwritten Digit Classifier**! This app uses a deep learning model to recognize handwritten digits from the famous MNIST dataset.

    ### 🤖 About the Model
    - **Model Architecture**: A Convolutional Neural Network (CNN) with two convolutional layers, max pooling, and fully connected layers.
    - **Training Accuracy**: 98% on the MNIST dataset.
    - **Test Accuracy**: 97% on unseen test data.

    The model is trained to classify handwritten digits (0-9) from 28x28 grayscale images.
    """)

    # Add a helpful link
    st.sidebar.write("📝 **Learn more about the MNIST dataset** [here](http://yann.lecun.com/exdb/mnist/).")

# "How to Use the App" section
if section == "🚀 How to Use the App":
    st.write("""
    ## 🚀 How to Use the App

    You can upload an image of a handwritten digit, and the app will predict which digit (0-9) it is. The model expects images that are similar to the MNIST dataset (28x28 grayscale images). 

    ### 📤 Upload Options
    1. **MNIST Dataset Sample Images**: Download sample images from the [official MNIST dataset](http://yann.lecun.com/exdb/mnist/).
    2. **Manually Draw a Digit**: Use an online tool like [Google's Quick, Draw!](https://quickdraw.withgoogle.com/) or [Autodraw](https://www.autodraw.com/) to draw your own digit and upload the image.
    3. **Create Your Own Image**: Use any drawing tool (like Paint) to create a simple handwritten digit on a white background.
    4. **Generate Images with Code**: Use Python code to generate digit images that mimic the MNIST dataset format.
    """)

# "Upload Image" section
if section == "🖼️ Upload Image":
    st.write("### Upload Your Handwritten Digit Image")
    
    # Load the pre-trained model
    model = load_model()
    
    # Function to process and predict from an image
    def predict(image, model):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # Preprocess image
        image = transform(image).unsqueeze(0)
        # Make prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            return predicted.item()

    # Upload image widget
    uploaded_file = st.file_uploader("Choose an image (jpg or png format)...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Open the image file
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.write("Classifying the digit...")

        # Predict the class of the image
        label = predict(image, model)

        # Enhanced result display
        st.markdown(f"<h1 style='text-align: center; color: green;'>Predicted Digit: {label}</h1>", unsafe_allow_html=True)

# Sidebar footer with contact info or helpful links
st.sidebar.markdown("---")
st.sidebar.write("🔗 **Helpful Links**")
st.sidebar.write("[MNIST Dataset](http://yann.lecun.com/exdb/mnist/)")
st.sidebar.write("[Streamlit Documentation](https://docs.streamlit.io/)")

st.sidebar.markdown("💻 Created with [Streamlit](https://streamlit.io).")
