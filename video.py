import os
import numpy as np
import threading
from PIL import Image, ImageTk
import tkinter as tk
import wave
import pyaudio
import time
import cv2

class VideoDisplay:
    def __init__(self, width=960, height=540, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.root = None
        self.panel = None
        self.is_playing = False
        self.current_frame = 0
        self.frames = []
        self.audio_thread = None
        self.audio_position = 0
        self.audio_file = None
        self.audio_stream = None
        self.audio_playback_lock = threading.Lock()
        self.frame_duration = 1 / fps
        self.start_time = None

    def read_raw_video_frames(self, video_path):
        try:
            frame_length = self.width * self.height * 3
            
            with open(video_path, 'rb') as f:
                video_data = f.read()
            
            total_frames = len(video_data) // frame_length
            frames = []

            for frame_index in range(total_frames):
                start = frame_index * frame_length
                frame_data = video_data[start:start + frame_length]

                r = np.frombuffer(frame_data[0::3], dtype=np.uint8).reshape((self.height, self.width))
                g = np.frombuffer(frame_data[1::3], dtype=np.uint8).reshape((self.height, self.width))
                b = np.frombuffer(frame_data[2::3], dtype=np.uint8).reshape((self.height, self.width))

                rgb_array = np.stack([r, g, b], axis=-1)
                
                # Convert RGB to YCrCb
                ycrcb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2YCrCb)
                
                # Convert YCrCb back to RGB
                rgb_array = cv2.cvtColor(ycrcb_array, cv2.COLOR_YCrCb2RGB)
                
                frames.append(Image.fromarray(rgb_array))
            
            return frames

        except Exception as e:
            print(f"Error reading video: {e}")
            return []

    def play_audio(self, audio_path):
        try:
            if not self.audio_file:
                self.audio_file = wave.open(audio_path, 'rb')
                self.audio_stream = pyaudio.PyAudio().open(
                    format=pyaudio.PyAudio().get_format_from_width(self.audio_file.getsampwidth()),
                    channels=self.audio_file.getnchannels(),
                    rate=44100,
                    output=True,
                )

            self.audio_file.setpos(self.audio_position)
            chunk_size = int(44100 * self.frame_duration)
            data = self.audio_file.readframes(chunk_size)

            while data and self.is_playing:
                with self.audio_playback_lock:
                    self.audio_stream.write(data)
                    self.audio_position = self.audio_file.tell()
                data = self.audio_file.readframes(chunk_size)

        except Exception as e:
            print(f"Error playing audio: {e}")

    def play_video(self, video_path, audio_path):
        self.frames = self.read_raw_video_frames(video_path)
        
        if not self.frames:
            print("Could not read video frames")
            return

        self.root = tk.Tk()
        self.root.title("Video Display (RGB via YCrCb)")

        self.panel = tk.Label(self.root)
        self.panel.pack(side="top", fill="both", expand="yes")

        self.play_pause_button = tk.Button(
            self.root, 
            text="Pause", 
            command=lambda: self.toggle_play_pause(audio_path)
        )
        self.play_pause_button.pack(side="bottom")

        self.is_playing = True
        self.start_time = time.time()
        self.audio_thread = threading.Thread(target=self.play_audio, args=(audio_path,))
        self.audio_thread.start()
        self.current_frame = 0

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.update_frame()

        self.root.mainloop()

    def update_frame(self):
        if not self.is_playing:
            return

        if self.current_frame <= len(self.frames):
            photo = ImageTk.PhotoImage(self.frames[self.current_frame])
            
            self.panel.config(image=photo)
            self.panel.image = photo

            if self.current_frame + 1 != len(self.frames):
                self.current_frame += 1


            current_time = time.time()
            next_frame_time = self.start_time + (self.current_frame * self.frame_duration)
            wait_time = max(int((next_frame_time - current_time) * 1000), 1)

            self.root.after(wait_time, self.update_frame)
        else:
            print("Video finished.")
            self.is_playing = False
            self.stop_playback()

    def stop_playback(self):
        self.is_playing = False

        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.audio_file:
            self.audio_file.close()
        self.root.quit()

    def on_closing(self):
        self.stop_playback()
        self.root.destroy()

    def toggle_play_pause(self, audio_path):
        with self.audio_playback_lock:
            self.is_playing = not self.is_playing

        if self.is_playing:
            self.play_pause_button.config(text="Pause")
            self.start_time = time.time() - (self.current_frame * self.frame_duration)
            self.audio_thread = threading.Thread(target=self.play_audio, args=(audio_path,))
            self.audio_thread.start()
            self.update_frame()
        else:
            self.play_pause_button.config(text="Play")

def main(args):
    if len(args) < 2:
        print("Using default video and audio files")
        audio_path = "WalkingStaticBackground.wav"
        video_path = "WalkingStaticBackground.rgb"
    else:
        video_path = args[0]
        audio_path = args[1]

    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist")
        return

    if not os.path.exists(audio_path):
        print(f"Audio file {audio_path} does not exist")
        return

    width = 960
    height = 540
    fps = 30

    video_display = VideoDisplay(width, height, fps)
    video_display.play_video(video_path, audio_path)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
