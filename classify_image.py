import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('image_classification_model.h5')

# CIFAR-10 class labels
class_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def preprocess_image(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(32, 32))
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to match the shape of training data
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image
    img_array /= 255.0
    
    return img_array

def classify_image(img_path):
    # Preprocess the image
    img_array = preprocess_image(img_path)
    
    # Make predictions
    predictions = model.predict(img_array)
    
    # Get the index of the highest probability
    predicted_class = np.argmax(predictions[0])
    
    return predicted_class, predictions[0][predicted_class]

def get_class_label(class_index):
    return class_labels[class_index]

def open_image():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename()
    
    # Load the image using PIL
    img = Image.open(file_path)
    img.thumbnail((200, 200))
    img = ImageTk.PhotoImage(img)
    
    # Display the image in the GUI
    image_label.config(image=img)
    image_label.image = img
    
    # Classify the image
    predicted_class, confidence = classify_image(file_path)
    class_label = get_class_label(predicted_class)
    
    # Display the classification result
    result_label.config(text=f'Predicted class: {class_label}')

# Create the main window
window = tk.Tk()
window.title('Image Classification')
window.geometry('400x400')

# Create a button to open an image
open_button = tk.Button(window, text='Open Image', command=open_image)
open_button.pack(pady=20)

# Create a label to display the image
image_label = Label(window)
image_label.pack()

# Create a label to display the classification result
result_label = Label(window, text='Predicted class: ', font=('Arial', 14))
result_label.pack(pady=20)

# Run the main loop
window.mainloop()

# with confidence: {confidence:.4f}