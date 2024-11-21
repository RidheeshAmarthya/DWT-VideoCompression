import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

#RIdheesh

class ImageDisplay:
    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height
        self.root = None
        self.panel = None

    def read_image_rgb(self, img_path):
        """
        Read a raw RGB image file where color channels are stored separately
        
        Args:
            img_path (str): Path to the raw RGB image file
        
        Returns:
            PIL.Image: Processed image
        """
        try:
            # Read the entire file
            with open(img_path, 'rb') as f:
                # Read the entire file content
                data = f.read()
                
            # Create numpy arrays for each color channel
            r = np.frombuffer(data[:self.width * self.height], dtype=np.uint8).reshape((self.height, self.width))
            g = np.frombuffer(data[self.width * self.height:2 * self.width * self.height], dtype=np.uint8).reshape((self.height, self.width))
            b = np.frombuffer(data[2 * self.width * self.height:], dtype=np.uint8).reshape((self.height, self.width))
            
            # Combine channels
            img_array = np.stack([r, g, b], axis=-1)
            
            # Convert to PIL Image
            return Image.fromarray(img_array)

        except Exception as e:
            print(f"Error reading image: {e}")
            return None

    def show_image(self, img_path):
        """
        Display the image in a Tkinter window
        
        Args:
            img_path (str): Path to the image file
        """
        # Read the image
        img = self.read_image_rgb(img_path)
        
        if img is None:
            print("Could not read the image")
            return

        # Create the main window
        self.root = tk.Tk()
        self.root.title("Image Display")

        # Convert PIL image to PhotoImage
        photo = ImageTk.PhotoImage(img)

        # Create label with the image
        self.panel = tk.Label(self.root, image=photo)
        self.panel.image = photo  # Keep a reference!
        self.panel.pack(side="top", fill="both", expand="yes")

        # Start the GUI event loop
        self.root.mainloop()

def main(args):
    if len(args) < 1:
        print("Please provide an image path")
        return

    # Check if file exists
    if not os.path.exists(args[0]):
        print(f"File {args[0]} does not exist")
        return

    image_display = ImageDisplay()
    image_display.show_image(args[0])

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])