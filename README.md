# Multimedia Systems Design Project

## Output
<img src="https://github.com/RidheeshAmarthya/RidheeshAmarthya/blob/main/wallpaper.gif](https://github.com/RidheeshAmarthya/portfolio-website/blob/main/assets/MM-F2.gif">
<img src="https://github.com/RidheeshAmarthya/RidheeshAmarthya/blob/main/wallpaper.gif](https://github.com/RidheeshAmarthya/portfolio-website/blob/main/assets/MM-F.gif">

## Foreground/Background segmented compression 

Developed an innovative approach to video compression by segmenting foreground and background regions using block-based motion detection techniques. Leveraged Discrete Cosine Transform (DCT) to efficiently compress video frames, applying differential compression rates to foreground (high-priority regions with significant motion) and background (static or less critical regions). This method significantly reduced bandwidth usage in video-based communication while maintaining high visual quality for dynamic content. The project demonstrated the practical integration of motion analysis and DCT-based compression in real-time applications like video conferencing and streaming.

The output is shown to have background to be highly quantized whereas the foreground isn't quantized. There's option to play, pause and replay in the custom GUI player.

## Method

- Image frame of size 960 x 540 divided into 16 x 16 macroblocks.
- Uses Three Step Search algorithm to find the motion vectors along X and Y direction, finds magnitude and classifies each macroblock as foreground or background as per the set threshold.
- Macroblocks undergo DCT conversion as per n1 (for foreground) and n2 (for background). For best results use n1 = 0 and n2 = 7.
- Audio is synced with the video playback using time.time()

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
