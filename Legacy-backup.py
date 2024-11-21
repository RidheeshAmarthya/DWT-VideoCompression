import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

class VideoDisplay:
    def __init__(self, width=512, height=512, fps=30):
        """
        Initialize video display parameters
        
        Args:
            width (int): Width of each video frame
            height (int): Height of each video frame
            fps (int): Frames per second for playback
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.root = None
        self.panel = None
        self.is_playing = False
        self.current_frame = 0

    def read_raw_video_frames(self, video_path):
        """
        Read raw RGB video frames from a file
        
        Args:
            video_path (str): Path to the raw video file
        
        Returns:
            list: List of PIL Image frames
        """
        try:
            # Calculate frame size
            frame_length = self.width * self.height * 3
            
            # Read the entire file
            with open(video_path, 'rb') as f:
                video_data = f.read()
            
            # Calculate number of frames
            total_frames = len(video_data) // frame_length
            frames = []

            # Extract and process each frame
            for frame_index in range(total_frames):
                start = frame_index * frame_length
                frame_data = video_data[start:start + frame_length]

                # Split frame data into channels
                r = np.frombuffer(frame_data[0::3], dtype=np.uint8).reshape((self.height, self.width))
                g = np.frombuffer(frame_data[1::3], dtype=np.uint8).reshape((self.height, self.width))
                b = np.frombuffer(frame_data[2::3], dtype=np.uint8).reshape((self.height, self.width))

                
                # Combine channels
                img_array = np.stack([r, g, b], axis=-1)
                
                # Convert to PIL Image
                frames.append(Image.fromarray(img_array))
            
            return frames

        except Exception as e:
            print(f"Error reading video: {e}")
            return []

    def play_video(self, video_path):
        """
        Play the video in a Tkinter window
        
        Args:
            video_path (str): Path to the raw video file
        """
        # Read video frames
        self.frames = self.read_raw_video_frames(video_path)
        
        if not self.frames:
            print("Could not read video frames")
            return

        # Create the main window
        self.root = tk.Tk()
        self.root.title("Video Display")

        # Create label to display frames
        self.panel = tk.Label(self.root)
        self.panel.pack(side="top", fill="both", expand="yes")

        # Add play/pause button
        self.play_pause_button = tk.Button(
            self.root, 
            text="Pause", 
            command=self.toggle_play_pause
        )
        self.play_pause_button.pack(side="bottom")

        # Start video playback
        self.is_playing = True
        self.current_frame = 0
        self.update_frame()

        # Start the GUI event loop
        self.root.mainloop()

    def update_frame(self):
        """
        Update the displayed frame and schedule next frame
        """
        if not self.is_playing:
            return

        # Get current frame
        if self.current_frame < len(self.frames):
            # Convert PIL image to PhotoImage
            photo = ImageTk.PhotoImage(self.frames[self.current_frame])
            
            # Update panel
            self.panel.config(image=photo)
            self.panel.image = photo  # Keep a reference!

            # Increment frame
            self.current_frame += 1

            # Schedule next frame
            self.root.after(int(1000 / self.fps), self.update_frame)
        else:
            # Loop back to start
            self.current_frame = 0
            self.update_frame()

    def toggle_play_pause(self):
        """
        Toggle play/pause state of the video
        """
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_pause_button.config(text="Pause")
            self.update_frame()
        else:
            self.play_pause_button.config(text="Play")

def main(args):
    if len(args) < 1:
        print("No input provided using default settings")
        input = "WalkingStaticBackground.rgb"
    else:
        input = args[0]

    # Determine video dimensions if provided
    width = 960
    height = 540
    fps = 30

    if len(args) > 1:
        width = int(args[1])
    if len(args) > 2:
        height = int(args[2])
    if len(args) > 3:
        fps = int(args[3])

    video_display = VideoDisplay(width, height, fps)
    video_display.play_video(input)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])