import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_file_structure(file_path, ax, is_compressed):
    file_size = os.path.getsize(file_path)
    width, height = 960, 540  # Based on the encoder's default dimensions

    if not is_compressed:
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black'))
        
        # RGB structure visualization
        ax.add_patch(plt.Rectangle((0, 0), 1/3, 1, fill=True, facecolor='red', alpha=0.3))
        ax.add_patch(plt.Rectangle((1/3, 0), 1/3, 1, fill=True, facecolor='green', alpha=0.3))
        ax.add_patch(plt.Rectangle((2/3, 0), 1/3, 1, fill=True, facecolor='blue', alpha=0.3))
        ax.text(1/6, 0.5, 'R', ha='center', va='center')
        ax.text(1/2, 0.5, 'G', ha='center', va='center')
        ax.text(5/6, 0.5, 'B', ha='center', va='center')
        
        # Show macro blocks
        macro_block_size = 16
        for i in range(0, width, macro_block_size):
            ax.axvline(x=i/width, color='white', linestyle='--', alpha=0.5)
        for i in range(0, height, macro_block_size):
            ax.axhline(y=i/height, color='white', linestyle='--', alpha=0.5)
    else:
        # Compressed structure visualization
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        n1, n2 = map(int, lines[0].split())
        frame_count = len(lines) - 1
        blocks_per_frame = len(lines[1].split()) // (3 * 64 + 1)  # 3 channels, 8x8 DCT, 1 block type
        
        # Calculate the compressed size ratio
        compressed_ratio = file_size / (width * height * 3)
        
        # Adjust the rectangle size based on the compressed ratio
        rect_height = compressed_ratio
        rect_width = compressed_ratio
        
        ax.add_patch(plt.Rectangle((0, 0), rect_width, rect_height, fill=False, edgecolor='black'))
        
        # Metadata section
        metadata_height = 0.3
        ax.add_patch(plt.Rectangle((0, rect_height), rect_width, metadata_height, fill=True, facecolor='lightgray'))
        
        # Add metadata text
        ax.text(rect_width/2, rect_height + 0.25*metadata_height, f'n1={n1}, n2={n2}', ha='center', va='center')
        ax.text(rect_width/2, rect_height + 0.15*metadata_height, f'Total Frames: {frame_count}', ha='center', va='center')
        ax.text(rect_width/2, rect_height + 0.05*metadata_height, f'Blocks per frame: {blocks_per_frame}', ha='center', va='center')
        
        # Analyze only the first frame
        first_frame_data = [float(x) for x in lines[1].split()]
        block_types = first_frame_data[::193]  # Every 193rd value is a block type
        foreground_ratio = sum(block_types) / len(block_types)
        
        # Visualize block types for the first frame
        block_type_img = np.array(block_types).reshape(int(np.sqrt(len(block_types))), -1)
        ax.imshow(block_type_img, aspect='auto', cmap='YlOrRd', extent=[0, rect_width, 0, rect_height*0.5])
        ax.text(rect_width/2, rect_height*0.55, 'First Frame Block Types', ha='center', va='center')
        ax.text(rect_width/2, rect_height*0.65, f'Foreground Ratio: {foreground_ratio:.2f}', ha='center', va='center')
        
        # DCT coefficient heatmap (for the first block of the first frame)
        dct_coeffs = np.array(first_frame_data[1:193]).reshape(3, 64)  # First block (R, G, B)
        ax.imshow(dct_coeffs, aspect='auto', cmap='coolwarm', extent=[0, rect_width, rect_height*0.5, rect_height])
        ax.text(rect_width/2, rect_height*0.95, 'DCT Coefficients (First Block)', ha='center', va='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1 + metadata_height if is_compressed else 1)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    ax.set_title(f"File: {os.path.basename(file_path)}\nDimensions: {width}x{height}\nSize: {file_size} bytes")

# Usage
rgb_file_path = 'WalkingStaticBackground.rgb'
cmp_file_path = 'WalkingStaticBackground.cmp'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

visualize_file_structure(rgb_file_path, ax1, is_compressed=False)
visualize_file_structure(cmp_file_path, ax2, is_compressed=True)

plt.tight_layout()
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# def visualize_file_structures():
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
#     # Common settings
#     blocks = 10
#     block_size = 0.8
#     gap = 0.2
    
#     # RGB File Structure
#     ax1.set_title("RGB File Structure (450MB)")
#     ax1.set_xlim(0, blocks * (block_size + gap))
#     ax1.set_ylim(0, 3)
#     ax1.set_yticks([])
#     ax1.set_xticks([])
    
#     for i in range(3):
#         for j in range(blocks):
#             color = ['red', 'green', 'blue'][i]
#             ax1.add_patch(plt.Rectangle((j * (block_size + gap), i), block_size, block_size, fill=True, color=color, alpha=0.5))
#             ax1.text(j * (block_size + gap) + block_size/2, i + block_size/2, ['R', 'G', 'B'][i], ha='center', va='center')
    
#     ax1.text(blocks * (block_size + gap) / 2, -0.5, "960 x 540 pixels, 24 bits per pixel", ha='center', va='center')
    
#     # CMP File Structure
#     ax2.set_title("CMP File Structure (900MB)")
#     ax2.set_xlim(0, blocks * (block_size + gap))
#     ax2.set_ylim(0, 6)
#     ax2.set_yticks([])
#     ax2.set_xticks([])
    
#     # Header
#     ax2.add_patch(plt.Rectangle((0, 5), blocks * (block_size + gap), block_size, fill=True, color='lightgray'))
#     ax2.text(blocks * (block_size + gap) / 2, 5 + block_size/2, "Header: n1 n2", ha='center', va='center')
    
#     for i in range(5):
#         for j in range(blocks):
#             if i == 0:
#                 color = 'lightblue'
#                 text = "Type"
#             elif i == 1:
#                 color = 'red'
#                 text = "R"
#             elif i == 2:
#                 color = 'green'
#                 text = "G"
#             elif i == 3:
#                 color = 'blue'
#                 text = "B"
#             else:
#                 color = 'white'
#                 text = "Extra"
            
#             ax2.add_patch(plt.Rectangle((j * (block_size + gap), i), block_size, block_size, fill=True, color=color))
#             ax2.text(j * (block_size + gap) + block_size/2, i + block_size/2, text, ha='center', va='center')
    
#     ax2.text(blocks * (block_size + gap) / 2, -0.5, "Each block: Type + Y/U/V DCT coefficients (64 each)", ha='center', va='center')
    
#     plt.tight_layout()
#     plt.show()

# visualize_file_structures()

