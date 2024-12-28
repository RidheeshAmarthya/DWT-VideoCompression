# Multimedia Systems Design Project

## Foreground/Background segmented compression 

Developed an innovative approach to video compression by segmenting foreground and background regions using block-based motion detection techniques. Leveraged Discrete Cosine Transform (DCT) to efficiently compress video frames, applying differential compression rates to foreground (high-priority regions with significant motion) and background (static or less critical regions). This method significantly reduced bandwidth usage in video-based communication while maintaining high visual quality for dynamic content. The project demonstrated the practical integration of motion analysis and DCT-based compression in real-time applications like video conferencing and streaming.

## Requirements

To run the code, you need the following dependencies installed:

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- os
- time
- threading
- wave
- pyaudio
- tkinter
- PIL

## Usage

```bash
python video_encoder.py <.rgb> n1 n2
```
```bash
python video_decoder.py <.cmp> <.wav>
```
