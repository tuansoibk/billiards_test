import cv2
import matplotlib.pyplot as plt

# Function to display images
def show_image(image, title):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()