import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


model = tf.keras.models.load_model("cnn_skin_lesion_high_acc.h5")
img_size = (128, 128)


root = tk.Tk()
root.title("Skin Lesion Prediction")
root.geometry("400x450")


img_label = tk.Label(root)
img_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14, "bold"))
result_label.pack(pady=10)


def load_image():
    global img_path
    img_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if img_path:
        img = Image.open(img_path)
        img.thumbnail((250, 250))
        img = ImageTk.PhotoImage(img)

        img_label.configure(image=img)
        img_label.image = img
        result_label.config(text="")


def predict_image():
    if not img_path:
        messagebox.showerror("Error", "Please select an image first")
        return

    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction >= 0.5:
        result = f"MALIGNANT\nConfidence: {prediction:.2f}"
        result_label.config(text=result, fg="red")
        messagebox.showwarning("Prediction Result", result)
    else:
        result = f"BENIGN\nConfidence: {1 - prediction:.2f}"
        result_label.config(text=result, fg="green")
        messagebox.showinfo("Prediction Result", result)

tk.Button(root, text="Select Image", command=load_image, width=20).pack(pady=10)
tk.Button(root, text="Predict", command=predict_image, width=20).pack(pady=10)


root.mainloop()
