import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

# Load MNIST dataset
mnist = datasets.MNIST(root='./data', train=True, download=True)

# Save a sample image
for i in range(10):
    image, label = mnist[i]
    image.save(f"mnist_digit_{label}.png")

