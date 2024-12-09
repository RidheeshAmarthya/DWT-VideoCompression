import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

class VideoEncoder:
    def _init_(self, width=960, height=540, macro_block_size=16, dct_block_size=8):
        self.width = width
        self.height = height
        self.macro_block_size = macro_block_size
        self.dct_block_size = dct_block_size
        self.motion_threshold = 2.0
        self.i_frame_interval = 30
        self.paused = False
        self.current_frame = 0

    def read_raw_video_frames(self, video_path):
        frame_length = self.width * self.height * 3
        frames = []
        yuv_frames = []
        with open(video_path, 'rb') as f:
            video_data = f.read()
        total_frames = len(video_data) // frame_length
        for frame_index in range(total_frames):
            start = frame_index * frame_length
            frame_data = video_data[start:start + frame_length]
            r = np.frombuffer(frame_data[0::3], dtype=np.uint8).reshape((self.height, self.width))
            g = np.frombuffer(frame_data[1::3], dtype=np.uint8).reshape((self.height, self.width))
            b = np.frombuffer(frame_data[2::3], dtype=np.uint8).reshape((self.height, self.width))
            rgb_frame = np.stack([r, g, b], axis=-1)
            yuv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2YUV)
            frames.append(rgb_frame)
            yuv_frames.append(yuv_frame)
        return frames, yuv_frames

    def compute_block_difference(self, block1, block2):
        return np.mean(np.abs(block1.astype(np.float32) - block2.astype(np.float32)))

    def find_motion_vector(self, current_block, previous_frame_y, y, x, search_range=16):
        height, width = previous_frame_y.shape
        min_diff = float('inf')
        motion_vector = (0, 0)
        step_size = search_range // 2
        center_y, center_x = y, x
        while step_size >= 1:
            best_y, best_x = center_y, center_x
            for dy in [-step_size, 0, step_size]:
                for dx in [-step_size, 0, step_size]:
                    new_y = center_y + dy
                    new_x = center_x + dx
                    if (0 <= new_y < height - self.macro_block_size and
                        0 <= new_x < width - self.macro_block_size):
                        reference_block = previous_frame_y[new_y:new_y + self.macro_block_size,
                                                          new_x:new_x + self.macro_block_size]
                        diff = self.compute_block_difference(current_block, reference_block)
                        if diff < min_diff:
                            min_diff = diff
                            best_y, best_x = new_y, new_x
            center_y, center_x = best_y, best_x
            step_size //= 2
        motion_vector = (y - center_y, x - center_x)  # Corrected motion vector calculation
        return motion_vector, min_diff

    def segment_frame(self, current_frame_yuv, previous_frame_yuv):
        height, width = current_frame_yuv.shape[:2]
        motion_vectors = np.zeros((height // self.macro_block_size, width // self.macro_block_size, 2), dtype=np.float32)
        block_types = np.zeros((height // self.macro_block_size, width // self.macro_block_size), dtype=np.uint8)
        current_y = current_frame_yuv[:, :, 0]
        previous_y = previous_frame_yuv[:, :, 0]
        for y in range(0, height - self.macro_block_size + 1, self.macro_block_size):
            for x in range(0, width - self.macro_block_size + 1, self.macro_block_size):
                current_block = current_y[y:y + self.macro_block_size, x:x + self.macro_block_size]
                motion_vector, _ = self.find_motion_vector(current_block, previous_y, y, x)
                block_y = y // self.macro_block_size
                block_x = x // self.macro_block_size
                motion_vectors[block_y, block_x] = motion_vector
                motion_magnitude = np.sqrt(motion_vector[0]*2 + motion_vector[1]*2)
                block_types[block_y, block_x] = 1 if motion_magnitude > self.motion_threshold else 0
        return block_types, motion_vectors

    def visualize_motion_vectors(self, frame, motion_vectors):
        plt.clf()
        plt.imshow(frame)
        height, width = motion_vectors.shape[:2]
        y, x = np.mgrid[0:height, 0:width]
        x = x * self.macro_block_size + self.macro_block_size // 2
        y = y * self.macro_block_size + self.macro_block_size // 2
        dx = motion_vectors[:, :, 1]
        dy = motion_vectors[:, :, 0]
        plt.quiver(x, y, -dx, -dy, color='r', angles='xy', scale_units='xy', scale=0.25)
        plt.title('Motion Vectors Visualization')
        plt.axis('off')


    def dct_2d(self, block):
        return cv2.dct(block.astype(np.float32))

    def quantize(self, dct_block, n):
        quantization_matrix = np.full((self.dct_block_size, self.dct_block_size), 2 ** n)
        return np.round(dct_block / quantization_matrix).astype(np.int16)

    def compress_frame(self, frame, block_types, n1, n2):
        height, width = frame.shape[:2]
        compressed_blocks = []
        for y in range(0, height - self.dct_block_size + 1, self.dct_block_size):
            for x in range(0, width - self.dct_block_size + 1, self.dct_block_size):
                macro_y = min(y // self.macro_block_size, block_types.shape[0] - 1)
                macro_x = min(x // self.macro_block_size, block_types.shape[1] - 1)
                is_foreground = block_types[macro_y, macro_x]
                block = frame[y:y + self.dct_block_size, x:x + self.dct_block_size]
                if block.shape[0] != self.dct_block_size or block.shape[1] != self.dct_block_size:
                    continue
                quantized_channels = []
                for channel in range(3):
                    dct_coeffs = self.dct_2d(block[:, :, channel])
                    n = n1 if is_foreground else n2
                    quantized = self.quantize(dct_coeffs, n)
                    quantized_channels.append(quantized)
                compressed_blocks.append((is_foreground, quantized_channels))
        return compressed_blocks

    def save_compressed_file(self, compressed_data, n1, n2, output_file):
        with open(output_file, 'w') as f:
            f.write(f'{n1} {n2}\n')
            for frame_blocks in compressed_data:
                for block_type, channels in frame_blocks:
                    f.write(f'{block_type}')
                    for channel in channels:
                        coeffs = channel.flatten()
                        f.write(' ' + ' '.join(map(str, coeffs)))
                    f.write('\n')

    def encode_video(self, input_file, n1, n2):
        frames, yuv_frames = self.read_raw_video_frames(input_file)
        compressed_frames = []

        root = tk.Tk()
        root.title("Video Player")

        fig, ax = plt.subplots(figsize=(12, 8))
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()

        def update_frame():
            if not self.paused and self.current_frame < len(frames):
                if self.current_frame == 0 or self.current_frame % self.i_frame_interval == 0:
                    block_types = np.ones((self.height // self.macro_block_size, self.width // self.macro_block_size), dtype=np.uint8)
                    motion_vectors = np.zeros((self.height // self.macro_block_size, self.width // self.macro_block_size, 2), dtype=np.float32)
                else:
                    block_types, motion_vectors = self.segment_frame(yuv_frames[self.current_frame], yuv_frames[self.current_frame-1])
                
                self.visualize_motion_vectors(frames[self.current_frame], motion_vectors)
                canvas.draw()
                
                compressed_frame = self.compress_frame(frames[self.current_frame], block_types, n1, n2)
                compressed_frames.append(compressed_frame)
                
                self.current_frame += 1
                root.after(100, update_frame)
            elif self.current_frame >= len(frames):
                output_file = input_file.rsplit('.', 1)[0] + '.cmp'
                self.save_compressed_file(compressed_frames, n1, n2, output_file)
                print(f"Compression complete. Output saved as {output_file}")
                root.quit()

        def toggle_pause():
            self.paused = not self.paused
            if not self.paused:
                update_frame()

        pause_button = tk.Button(root, text="Pause/Resume", command=toggle_pause)
        pause_button.pack()

        update_frame()
        root.mainloop()

def main():
    if len(sys.argv) != 4:
        print("Usage: python encoder.py input_video.rgb n1 n2")
        return

    input_file = sys.argv[1]
    n1 = int(sys.argv[2])
    n2 = int(sys.argv[3])

    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return

    encoder = VideoEncoder()
    encoder.encode_video(input_file, n1, n2)

if __name__ == "_main_":
    main()