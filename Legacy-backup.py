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



    # import os
# import numpy as np
# import threading
# from PIL import Image, ImageTk
# import tkinter as tk
# import wave
# import pyaudio


# class VideoDisplay:
#     def __init__(self, width=512, height=512, fps=30):
#         """
#         Initialize video display parameters
        
#         Args:
#             width (int): Width of each video frame
#             height (int): Height of each video frame
#             fps (int): Frames per second for playback
#         """
#         self.width = width
#         self.height = height
#         self.fps = fps
#         self.root = None
#         self.panel = None
#         self.is_playing = False
#         self.current_frame = 0
#         self.frames = []
#         self.audio_thread = None
#         self.audio_position = 0  # Track the current audio position
#         self.audio_file = None
#         self.audio_stream = None
#         self.audio_playback_lock = threading.Lock()

#     def read_raw_video_frames(self, video_path):
#         """
#         Read raw RGB video frames from a file
        
#         Args:
#             video_path (str): Path to the raw video file
        
#         Returns:
#             list: List of PIL Image frames
#         """
#         try:
#             # Calculate frame size
#             frame_length = self.width * self.height * 3
            
#             # Read the entire file
#             with open(video_path, 'rb') as f:
#                 video_data = f.read()
            
#             # Calculate number of frames
#             total_frames = len(video_data) // frame_length
#             frames = []

#             # Extract and process each frame
#             for frame_index in range(total_frames):
#                 start = frame_index * frame_length
#                 frame_data = video_data[start:start + frame_length]

#                 # Split frame data into channels
#                 r = np.frombuffer(frame_data[0::3], dtype=np.uint8).reshape((self.height, self.width))
#                 g = np.frombuffer(frame_data[1::3], dtype=np.uint8).reshape((self.height, self.width))
#                 b = np.frombuffer(frame_data[2::3], dtype=np.uint8).reshape((self.height, self.width))

#                 # Combine channels
#                 img_array = np.stack([r, g, b], axis=-1)
                
#                 # Convert to PIL Image
#                 frames.append(Image.fromarray(img_array))
            
#             return frames

#         except Exception as e:
#             print(f"Error reading video: {e}")
#             return []

#     def play_audio(self, audio_path):
#         """
#         Play the audio file using PyAudio
        
#         Args:
#             audio_path (str): Path to the audio file
#         """
#         try:
#             # If this is the first time, load the audio file
#             if not self.audio_file:
#                 self.audio_file = wave.open(audio_path, 'rb')
#                 self.audio_stream = pyaudio.PyAudio().open(
#                     format=pyaudio.PyAudio().get_format_from_width(self.audio_file.getsampwidth()),
#                     channels=self.audio_file.getnchannels(),
#                     rate = 44100,
#                     # rate=self.audio_file.getframerate(),
#                     output=True,
#                 )

#             # Resume from the current position
#             self.audio_file.setpos(self.audio_position)
#             chunk_size = 1024
#             data = self.audio_file.readframes(chunk_size)

#             # Playback audio while maintaining lock
#             while data and self.is_playing:
#                 with self.audio_playback_lock:
#                     self.audio_stream.write(data)
#                     self.audio_position = self.audio_file.tell()  # Update current position
#                 data = self.audio_file.readframes(chunk_size)

#         except Exception as e:
#             print(f"Error playing audio: {e}")

#     def play_video(self, video_path, audio_path):
#         """
#         Play the video and synchronize with audio
        
#         Args:
#             video_path (str): Path to the raw video file
#             audio_path (str): Path to the audio file
#         """
#         # Read video frames
#         self.frames = self.read_raw_video_frames(video_path)
        
#         if not self.frames:
#             print("Could not read video frames")
#             return

#         # Create the main window
#         self.root = tk.Tk()
#         self.root.title("Video Display")

#         # Create label to display frames
#         self.panel = tk.Label(self.root)
#         self.panel.pack(side="top", fill="both", expand="yes")

#         # Add play/pause button
#         self.play_pause_button = tk.Button(
#             self.root, 
#             text="Pause", 
#             command=lambda: self.toggle_play_pause(audio_path)
#         )
#         self.play_pause_button.pack(side="bottom")

#         # Start video playback
#         self.is_playing = True
#         self.audio_thread = threading.Thread(target=self.play_audio, args=(audio_path,))
#         self.audio_thread.start()
#         self.current_frame = 0
#         self.update_frame()

#         # Start the GUI event loop
#         self.root.mainloop()

#     def update_frame(self):
#         """
#         Update the displayed frame and schedule next frame
#         """
#         if not self.is_playing:
#             return

#         # Get current frame
#         if self.current_frame < len(self.frames):
#             # Convert PIL image to PhotoImage
#             photo = ImageTk.PhotoImage(self.frames[self.current_frame])
            
#             # Update panel
#             self.panel.config(image=photo)
#             self.panel.image = photo  # Keep a reference!

#             # Increment frame
#             self.current_frame += 1

#             # Schedule next frame
#             self.root.after(int(1000 / self.fps), self.update_frame)
#         else:
#             print("Video finished.")
#             self.is_playing = False
#             self.audio_stream.stop_stream()
#             self.audio_stream.close()
#             self.audio_file.close()
#             self.root.destroy()
#             return

#     def toggle_play_pause(self, audio_path):
#         """
#         Toggle play/pause state of the video and audio
#         """
#         with self.audio_playback_lock:
#             self.is_playing = not self.is_playing

#         if self.is_playing:
#             self.play_pause_button.config(text="Pause")
#             # Restart the audio thread if paused
#             self.audio_thread = threading.Thread(target=self.play_audio, args=(audio_path,))
#             self.audio_thread.start()
#             self.update_frame()
#         else:
#             self.play_pause_button.config(text="Play")


# def main(args):
#     if len(args) < 2:
#         print("Using default video and audio files")
#         audio_path = "WalkingStaticBackground.wav"
#         video_path = "WalkingStaticBackground.rgb"
#     else:
#         audio_path = args[1]
#         video_path = args[0]

#     if not os.path.exists(video_path):
#         print(f"Video file {video_path} does not exist")
#         return

#     if not os.path.exists(audio_path):
#         print(f"Audio file {audio_path} does not exist")
#         return

#     # Determine video dimensions if provided
#     width = 960
#     height = 540

#     #This does not work. Regardless of the FPS set it always plays at 30fps, which is what we want so, it works!
#     fps = 30

#     if len(args) > 2:
#         width = int(args[2])
#     if len(args) > 3:
#         height = int(args[3])
#     if len(args) > 4:
#         fps = int(args[4])
#     else:
#         print("Using Default Settings")


#     video_display = VideoDisplay(width, height, fps)
#     video_display.play_video(video_path, audio_path)

# if __name__ == "__main__":
#     import sys
#     main(sys.argv[1:])
