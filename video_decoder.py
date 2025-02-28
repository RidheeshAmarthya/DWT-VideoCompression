import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
import wave
import pyaudio
import time
import sys


class VideoDecoder:
    def __init__(self, width=960, height=540, dct_block_size=8):
        self.width = width
        self.height = height
        self.dct_block_size = dct_block_size
        self.frames = []

    def parse_compressed_file(self, filepath):
        frames_data = []
        current_frame = []
        
        with open(filepath, 'r') as f:
            self.n1, self.n2 = map(int, f.readline().strip().split())
            
            for line in f:
                data = list(map(int, line.strip().split()))
                block_type = data[0]
                coeffs = data[1:]
                
                r_coeffs = coeffs[0:64]
                g_coeffs = coeffs[64:128]
                b_coeffs = coeffs[128:192]
                
                current_frame.append((block_type, r_coeffs, g_coeffs, b_coeffs))
                
                if len(current_frame) == (self.width // self.dct_block_size) * (self.height // self.dct_block_size):
                    frames_data.append(current_frame)
                    current_frame = []
                    
        return frames_data

    def dequantize(self, quantized_coeffs, block_type):
        n = self.n1 if block_type == 1 else self.n2
        quantization_matrix = (1 << n)  # Replace np.full with this scalar multiplication
        dequantized = np.array(quantized_coeffs).reshape((8, 8)) * quantization_matrix
        return dequantized.astype(np.float32)


    def reconstruct_frame(self, frame_blocks):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        block_idx = 0
        for y in range(0, self.height, self.dct_block_size):
            for x in range(0, self.width, self.dct_block_size):
                if block_idx >= len(frame_blocks):
                    break
                    
                block_type, r_coeffs, g_coeffs, b_coeffs = frame_blocks[block_idx]
                
                r_block = cv2.idct(self.dequantize(r_coeffs, block_type))
                g_block = cv2.idct(self.dequantize(g_coeffs, block_type))
                b_block = cv2.idct(self.dequantize(b_coeffs, block_type))
                
                r_block = np.clip(r_block, 0, 255).astype(np.uint8)
                g_block = np.clip(g_block, 0, 255).astype(np.uint8)
                b_block = np.clip(b_block, 0, 255).astype(np.uint8)

                frame[y:y + self.dct_block_size, x:x + self.dct_block_size, 0] = r_block
                frame[y:y + self.dct_block_size, x:x + self.dct_block_size, 1] = g_block
                frame[y:y + self.dct_block_size, x:x + self.dct_block_size, 2] = b_block #idhar blur lagaya tha
                
                block_idx += 1
                
        return frame

class VideoPlayer:
    def __init__(self, width=960, height=540, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_duration = 1 / fps
        self.is_playing = False
        self.current_frame = 0
        self.audio_position = 0
        self.audio_file = None
        self.audio_stream = None
        self.audio_playback_lock = threading.Lock()
        self.start_time = time.time()
        
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

            chunk_size = int(44100 * self.frame_duration * 2)  # Double the size
            self.audio_file.setpos(self.audio_position)
            data = self.audio_file.readframes(chunk_size)

            while data and self.is_playing:
                with self.audio_playback_lock:
                    self.audio_stream.write(data)
                    self.audio_position = self.audio_file.tell()
                data = self.audio_file.readframes(chunk_size)

        except Exception as e:
            print(f"Error playing audio: {e}")

    def update_frame(self):
        if not self.is_playing:
            return
        
        if self.current_frame <= len(self.frames):
            photo = ImageTk.PhotoImage(Image.fromarray(self.frames[self.current_frame]))
            self.panel.config(image=photo)
            self.panel.image = photo

            if self.current_frame + 1 != len(self.frames):
                self.current_frame += 1 #gadbad
            
            if self.current_frame == len(self.frames):
                self.is_playing = False
                self.play_pause_button.config(text="Play")
                return
            
            current_time = time.time()
            next_frame_time = self.start_time + (self.current_frame * self.frame_duration)
            wait_time = max(int((next_frame_time - current_time) * 1000), 1)
            self.root.after(wait_time, self.update_frame)
        else:
            self.stop_playback()


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

    def stop_playback(self):
        self.is_playing = False
        if self.audio_stream:
            try:
                if self.audio_stream.is_active():
                    self.audio_stream.stop_stream()
                self.audio_stream.close()
            except OSError as e:
                print(f"Error stopping audio stream: {e}")
        if self.audio_file:
            try:
                self.audio_file.close()
            except Exception as e:
                print(f"Error closing audio file: {e}")
        self.play_pause_button.config(text="Play")



    def replay(self, audio_path):
        if self.is_playing:
            self.toggle_play_pause(audio_path)
        else:
            self.stop_playback()
        self.current_frame = 0
        self.audio_position = 0
        self.audio_file = None
        self.audio_stream = None
        self.start_time = time.time()
        self.toggle_play_pause(audio_path)


    def play_video(self, frames, audio_path):
        self.frames = frames
        self.root = tk.Tk()
        self.root.title("Decoded Video Player")

        self.panel = tk.Label(self.root)
        self.panel.pack(side="top", fill="both", expand="yes")

        button_frame = tk.Frame(self.root)
        button_frame.pack(side="bottom")

        self.play_pause_button = tk.Button(
            button_frame, 
            text="Pause",
            command=lambda: self.toggle_play_pause(audio_path)
        )
        self.play_pause_button.pack(side="left", padx=5)

        self.replay_button = tk.Button(
            button_frame,
            text="Replay",
            command=lambda: self.replay(audio_path)
        )
        self.replay_button.pack(side="left", padx=5)

        self.is_playing = True
        self.audio_thread = threading.Thread(target=self.play_audio, args=(audio_path,))
        self.audio_thread.start()

        self.update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self.root.quit)
        self.root.mainloop()

def main():
    #count the time in seconds
    start_time = time.time()

    if len(sys.argv) != 3:
        print("Usage: python decoder.py input_video.cmp input_audio.wav")
        return

    cmp_file = sys.argv[1]
    audio_file = sys.argv[2]

    decoder = VideoDecoder()
    frames_data = decoder.parse_compressed_file(cmp_file)
    decoded_frames = [decoder.reconstruct_frame(frame_blocks) for frame_blocks in frames_data]

    player = VideoPlayer()
    player.play_video(decoded_frames, audio_file)

    print(f"Decoding took {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
