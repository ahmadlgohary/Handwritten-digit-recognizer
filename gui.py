from keras.models import load_model
from tkinter import *
import tkinter as tk
from PIL import Image
import numpy as np
import keras
model = load_model('handwritten.model')
from io import BytesIO
def predict_digit(img):
    
    img = img.resize((28,28))                   #resize image to 28x28 pixels
    img = img.convert('L')                      #convert rgb to grayscale
    img = np.invert(np.array(img))              #invert the image  
    img = img.reshape(-1,28,28,1)               #reshaping the array to fit the CNN input
    img = keras.utils.normalize(img, axis =1)   #normalize greyscale from 0-255 to 0-1 
    prediction = model.predict(img)             #passing the image as input to the CNN
    return np.argmax(prediction), max(prediction)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognise", command = self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
    def clear_all(self):
        self.canvas.delete("all")
    def classify_handwriting(self):
        ps = self.canvas.postscript() 
        # use PIL to convert to PNG 
        im = Image.open(BytesIO(ps.encode('utf-8')))
        digit, acc = predict_digit(im)
        digit =  digit.item()
        acc = max(acc)
        acc =  acc.item()
        self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
app = App()
mainloop()
