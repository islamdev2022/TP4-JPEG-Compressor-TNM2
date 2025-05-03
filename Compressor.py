import numpy as np
from PIL import Image, ImageTk
from Huffman import process_huffman_encoding,huffman_decode
import os
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
import numpy as np
def rgb_to_ycbcr(image):

    transform_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])

    img_array = np.array(image, dtype=np.float32)
    R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

    Y = transform_matrix[0, 0] * R + transform_matrix[0, 1] * G + transform_matrix[0, 2] * B
    Cb = transform_matrix[1, 0] * R + transform_matrix[1, 1] * G + transform_matrix[1, 2] * B + 128
    Cr = transform_matrix[2, 0] * R + transform_matrix[2, 1] * G + transform_matrix[2, 2] * B + 128

    return Y, Cb, Cr



def ycbcr_to_rgb(Y, Cb, Cr):
    """Convert YCbCr back to RGB color space"""

    Cb , Cr = upsample(Cb, Cr, "3")
    print("CB SHAPE",Cb.shape)
    print("CR SHAPE",Cr.shape)

    # Assume Cb and Cr are in [0, 255] and need to be centered
    Cb -= 128.0
    Cr -= 128.0

    inv_transform = np.array([
            [1, 0, 1.402],
            [1, -0.344136, -0.714136],
            [1, 1.772, 0]
        ])

    R = inv_transform[0, 0] * Y + inv_transform[0, 2] * Cr
    G = inv_transform[1, 0] * Y + inv_transform[1, 1] * Cb + inv_transform[1, 2] * Cr
    B = inv_transform[2, 0] * Y + inv_transform[2, 1] * Cb
    rgb_image = np.clip(np.stack((R, G, B), axis=-1), 0, 255).astype(np.uint8)
    return rgb_image

    
def psnr(original, compressed):
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
        return 10 * np.log10(255 ** 2 / mse) if mse != 0 else float('inf')

    
def blockwise_transform(img, block_size, transform_func, D=None):
    h, w = img.shape
    transformed_img = np.zeros_like(img, dtype=np.float32)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if D is not None:
                transformed_img[i:i + block_size, j:j + block_size] = transform_func(
                    img[i:i + block_size, j:j + block_size], D)
            else:
                transformed_img[i:i + block_size, j:j + block_size] = transform_func(
                    img[i:i + block_size, j:j + block_size])

    return transformed_img

def dct_matrix(N):
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == 0:
                D[i, j] = np.sqrt(1 / N)
            else:
                D[i, j] = np.sqrt(2 / N) * np.cos((np.pi * (2 * j + 1) * i) / (2 * N))
    return D


def dct_2d_matrix(block, D):
    temp = np.dot(D, block)
    return np.dot(temp, D.T)

def idct_2d_matrix(dct_block, D):
    temp = np.dot(D.T, dct_block)
    return np.dot(temp, D)

def pad_image_to_block_size(image, block_size):
    h, w = image.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h != 0 or pad_w != 0:
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
        return padded_image, (pad_h, pad_w)
    return image, (0, 0)

def unpad_image(image, padding):
    pad_h, pad_w = padding
    if pad_h != 0 or pad_w != 0:
        return image[:-pad_h, :-pad_w] if pad_h !=0 and pad_w !=0 else \
               image[:-pad_h, :] if pad_h !=0 else \
               image[:, :-pad_w]
    return image

def downsample(Y, Cb, Cr, mode):    
    if mode == "4:2:2":
        # Downsample Cb/Cr horizontally (average 2x1 blocks)
        Cb = (Cb[:, ::2] + Cb[:, 1::2]) / 2
        Cr = (Cr[:, ::2] + Cr[:, 1::2]) / 2
    elif mode == "4:2:0":
        # Downsample Cb/Cr both axes (average 2x2 blocks)
        Cb = (Cb[::2, ::2] + Cb[1::2, ::2] + Cb[::2, 1::2] + Cb[1::2, 1::2]) / 4
        Cr = (Cr[::2, ::2] + Cr[1::2, ::2] + Cr[::2, 1::2] + Cr[1::2, 1::2]) / 4
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    return Y, Cb, Cr

def upsample( Cb, Cr, mode):
        """Upsample Cb and Cr components back to original size"""
        if mode == "2":  # 4:2:2
            Cb = np.repeat(Cb, 2, axis=1)
            Cr = np.repeat(Cr, 2, axis=1)
        elif mode == "3":  # 4:2:0
            Cb = np.repeat(np.repeat(Cb, 2, axis=0), 2, axis=1)
            Cr = np.repeat(np.repeat(Cr, 2, axis=0), 2, axis=1)
        return Cb, Cr
    
def get_quantization_tables(quality):
    """Generate quantization tables based on quality (0-50)"""
    # Standard quantization tables (baseline)
    Q_Y_base = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)

    Q_C_base = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ], dtype=np.float32)

    # Ensure quality is in valid range
    quality = max(0, min(50, quality))
    
    # Convert inverse quality scale to the standard JPEG quality factor
    # where quality=0 → Q=100, quality=50 → Q=1
    standard_q = 100 - (quality * 2)
    
    if standard_q <= 0:  # This would happen if quality = 50
        standard_q = 1  # Set to minimum quality

    # Now use the standard formula with the converted quality value
    if standard_q < 50:
        scale_factor = 5000 / standard_q
    elif standard_q < 100:
        scale_factor = 200 - 2 * standard_q
    else:  # standard_q == 100
        scale_factor = 1

    # Apply scaling
    scale_factor = scale_factor / 100.0
    
    Q_Y = np.clip(np.round(Q_Y_base * scale_factor), 1, 255)
    Q_C = np.clip(np.round(Q_C_base * scale_factor), 1, 255)
    
    return Q_Y, Q_C
def quantize_dct(dct_coeffs, quantization_table):
    """Quantize DCT coefficients using the given quantization table"""
    h, w = dct_coeffs.shape
    quantized = np.zeros_like(dct_coeffs)
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = dct_coeffs[i:i+8, j:j+8]
            quantized[i:i+8, j:j+8] = np.round(block / quantization_table)
    
    return quantized

def make_dimensions_divisible_by_block_size(img, block_size):
    """Adjust image dimensions to be divisible by block_size"""
    h, w = img.shape[:2]
    new_h = (h // block_size) * block_size
    new_w = (w // block_size) * block_size
    return img[:new_h, :new_w]



def log_scale_dct(dct_coeffs):
    # Take absolute value (DCT can have negative values)
    abs_dct = np.abs(dct_coeffs)
    # Add 1 to avoid log(0)
    log_dct = np.log(1 + abs_dct)
    # Normalize to 0-255 for display
    return (log_dct / np.max(log_dct) * 255).astype(np.uint8)


def dequantize_dct(quantized_dct, quantization_table):
    """Reverse the quantization process blockwise"""
    h, w = quantized_dct.shape
    block_size = quantization_table.shape[0]  # Usually 8x8
    
    # Create output array of the same size
    dequantized = np.zeros_like(quantized_dct, dtype=float)
    
    # Process each block
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Get current block
            block = quantized_dct[i:i+block_size, j:j+block_size]
            # Apply dequantization (element-wise multiplication with quantization table)
            dequantized_block = block * quantization_table
            # Store back
            dequantized[i:i+block_size, j:j+block_size] = dequantized_block
            
    return dequantized

def zigzag_scan(matrix):
    rows, cols = len(matrix), len(matrix[0])
    result = []
    for s in range(rows + cols - 1):
        if s % 2 == 0:
            # Even diagonals: top-right to bottom-left (row-major)
            for i in range(max(0, s - rows + 1), min(s + 1, cols)):
                x = s - i
                y = i
                result.append(matrix[x][y])
        else:
            # Odd diagonals: bottom-left to top-right (row-major)
            for i in range(max(0, s - cols + 1), min(s + 1, rows)):
                x = i
                y = s - i
                result.append(matrix[x][y])
    return result

def Run_Length(data):
    rle = []
    zero_count = 0

    for coeff in data:
        if coeff == 0:
            zero_count += 1
        else:
            while zero_count > 15:
                rle.append((15, 0))  # JPEG rule: max zero run is 15
                zero_count -= 16
            rle.append((zero_count, coeff))
            zero_count = 0

    rle.append((0, 0))  # EOB marker
    return rle


def inverse_zigzag(zigzag_data):
    """
    Convert zigzag format back to 8x8 blocks
    """
    # Get the number of coefficients per block (should be 64 for 8x8 blocks)
    coeffs_per_block = 64
    total_coeffs = len(zigzag_data)
    
    # Calculate number of complete blocks
    num_blocks = total_coeffs // coeffs_per_block
    print(f"Creating {num_blocks} blocks from {total_coeffs} coefficients")
    
    # Zigzag pattern for 8x8 blocks
    zigzag_pattern = [
        (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
        (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
        (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
        (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
        (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
        (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
        (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
        (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
    ]
    
    blocks = []
    for i in range(num_blocks):
        # Create an empty 8x8 block
        block = np.zeros((8, 8), dtype=np.float32)
        
        # Fill the block following the zigzag pattern
        for j in range(coeffs_per_block):
            idx = i * coeffs_per_block + j
            if idx < total_coeffs:
                row, col = zigzag_pattern[j]
                block[row, col] = zigzag_data[idx]
        
        blocks.append(block)
    
    print(f"Created {len(blocks)} blocks from zigzag data")
    return np.array(blocks)

def reconstruct_full_zigzag(dc_values, ac_flat, block_size=64):
    full_zigzag = []
    ac_per_block = block_size - 1  # 63 AC coefficients per block
    for i, dc in enumerate(dc_values):
        start = i * ac_per_block
        end = start + ac_per_block
        ac = ac_flat[start:end]
        full_zigzag.append(dc)  # Add DC
        full_zigzag.extend(ac)  # Add AC
    return full_zigzag

def flatten_rle(rle):
    return [f"{run}/{val}" for run, val in rle]

def unflatten_rle(decoded_string):
    """Parse RLE data using the delimiter."""
    rle_data = []
    symbols = decoded_string.split("|")  # Split by delimiter
    for symbol in symbols:
        if "/" in symbol:
            run, val = symbol.split("/", 1)
            try:
                run_int = int(run)
                val_float = float(val)
                rle_data.append((run_int, val_float))
            except ValueError:
                print(f"Invalid symbol: {symbol}")
    return rle_data

def inverse_run_length(rle_data):
    all_coefficients = []
    current_block = []
    for run, val in rle_data:
        if (run, val) == (0, 0):  # EOB marker
            # Pad to 63 coefficients for this block
            remaining = 63 - len(current_block)
            current_block.extend([0] * remaining)
            all_coefficients.extend(current_block)
            current_block = []
        else:
            current_block.extend([0] * run)
            current_block.append(val)
    # Handle last block if no EOB
    if current_block:
        remaining = 63 - len(current_block)
        current_block.extend([0] * remaining)
        all_coefficients.extend(current_block)
    return all_coefficients

# Alternative temporary solution if parsing fails completely
def safe_zigzag_data():
    """
    Create safe zigzag data as a fallback
    Returns a list of 64 zeros (8x8 block)
    """
    return [0] * 64

# Set the appearance mode and default color theme
ctk.set_appearance_mode("System")  
ctk.set_default_color_theme("blue")

class JPEGCompressorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("JPEG Compressor")
        self.geometry("1200x800")
        
        # Initialize variables
        self.image_path = None
        self.original_image = None
        self.compressed_image = None
        self.original_size = 0
        self.compressed_size = 0
        self.compression_rate = 0
        self.psnr_value = 0
        
        # Zoom variables
        self.original_zoom_factor = 1.0
        self.compressed_zoom_factor = 1.0
        self.original_img_np = None  # Store original image as numpy array
        self.compressed_img_np = None  # Store compressed image as numpy array
        
        # Create the main frame with two columns
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Left frame for controls
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Configure left frame grid
        self.left_frame.grid_columnconfigure(0, weight=1)
        
        # Right frame for image display
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Configure right frame grid
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=1)
        
        # Create controls in left frame
        self.create_controls()
        
        # Create image display areas in right frame
        self.create_image_display()

    def create_controls(self):
        # Title
        title_label = ctk.CTkLabel(self.left_frame, text="JPEG Compressor", font=ctk.CTkFont(size=24, weight="bold"))
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Image selection button
        self.select_button = ctk.CTkButton(self.left_frame, text="Select Image", command=self.select_image)
        self.select_button.grid(row=1, column=0, padx=20, pady=10)
        
        # Image path display
        self.path_label = ctk.CTkLabel(self.left_frame, text="No image selected", wraplength=300)
        self.path_label.grid(row=2, column=0, padx=20, pady=5)
        
        # Original image details
        self.original_details_frame = ctk.CTkFrame(self.left_frame)
        self.original_details_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.original_details_frame, text="Original Image:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.original_size_label = ctk.CTkLabel(self.original_details_frame, text="Size: N/A")
        self.original_size_label.grid(row=1, column=0, padx=10, pady=2, sticky="w")
        self.original_dims_label = ctk.CTkLabel(self.original_details_frame, text="Dimensions: N/A")
        self.original_dims_label.grid(row=2, column=0, padx=10, pady=2, sticky="w")
        
        # Quality slider
        quality_frame = ctk.CTkFrame(self.left_frame)
        quality_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(quality_frame, text="Quality Factor (0=best, 50=worst):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.quality_slider = ctk.CTkSlider(quality_frame, from_=0, to=50, number_of_steps=50)
        self.quality_slider.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.quality_slider.set(25)  # Default middle value
        
        self.quality_value_label = ctk.CTkLabel(quality_frame, text="25")
        self.quality_value_label.grid(row=1, column=1, padx=10, pady=5)
        
        # Update quality value when slider changes
        self.quality_slider.configure(command=self.update_quality_value)
        
        # Compress button
        self.compress_button = ctk.CTkButton(self.left_frame, text="Compress Image", command=self.compress_image)
        self.compress_button.grid(row=5, column=0, padx=20, pady=10)
        self.compress_button.configure(state="disabled")
        
        # Compression results frame
        self.results_frame = ctk.CTkFrame(self.left_frame)
        self.results_frame.grid(row=6, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.results_frame, text="Compression Results:", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.compressed_size_label = ctk.CTkLabel(self.results_frame, text="Compressed Size: N/A")
        self.compressed_size_label.grid(row=1, column=0, padx=10, pady=2, sticky="w")
        
        self.compression_rate_label = ctk.CTkLabel(self.results_frame, text="Compression Rate: N/A")
        self.compression_rate_label.grid(row=2, column=0, padx=10, pady=2, sticky="w")
        
        self.psnr_label = ctk.CTkLabel(self.results_frame, text="PSNR: N/A")
        self.psnr_label.grid(row=3, column=0, padx=10, pady=2, sticky="w")
        
        # Save button
        self.save_button = ctk.CTkButton(self.left_frame, text="Save Compressed Image", command=self.save_compressed_image)
        self.save_button.grid(row=7, column=0, padx=20, pady=10)
        self.save_button.configure(state="disabled")
        
       # Zoom controls frame
        zoom_frame = ctk.CTkFrame(self.left_frame)
        zoom_frame.grid(row=8, column=0, padx=20, pady=10, sticky="ew")

        ctk.CTkLabel(zoom_frame, text="Zoom Controls:", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # Place the controls side by side (compressed on left, original on right)
        # Compressed image zoom controls - LEFT SIDE
        ctk.CTkLabel(zoom_frame, text="Compressed Image:").grid(row=1, column=0, padx=10, pady=5, sticky="w")

        comp_zoom_controls = ctk.CTkFrame(zoom_frame)
        comp_zoom_controls.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        self.comp_zoom_out_btn = ctk.CTkButton(comp_zoom_controls, text="-", width=30, command=self.compressed_zoom_out)
        self.comp_zoom_out_btn.grid(row=0, column=0, padx=5, pady=5)

        self.comp_zoom_reset_btn = ctk.CTkButton(comp_zoom_controls, text="Reset", width=60, command=self.compressed_zoom_reset)
        self.comp_zoom_reset_btn.grid(row=0, column=1, padx=5, pady=5)

        self.comp_zoom_in_btn = ctk.CTkButton(comp_zoom_controls, text="+", width=30, command=self.compressed_zoom_in)
        self.comp_zoom_in_btn.grid(row=0, column=2, padx=5, pady=5)

        self.comp_zoom_label = ctk.CTkLabel(comp_zoom_controls, text="100%")
        self.comp_zoom_label.grid(row=0, column=3, padx=10, pady=5)

        # Original image zoom controls - RIGHT SIDE
        ctk.CTkLabel(zoom_frame, text="Original Image:").grid(row=1, column=1, padx=10, pady=5, sticky="w")

        orig_zoom_controls = ctk.CTkFrame(zoom_frame)
        orig_zoom_controls.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        self.orig_zoom_out_btn = ctk.CTkButton(orig_zoom_controls, text="-", width=30, command=self.original_zoom_out)
        self.orig_zoom_out_btn.grid(row=0, column=0, padx=5, pady=5)

        self.orig_zoom_reset_btn = ctk.CTkButton(orig_zoom_controls, text="Reset", width=60, command=self.original_zoom_reset)
        self.orig_zoom_reset_btn.grid(row=0, column=1, padx=5, pady=5)

        self.orig_zoom_in_btn = ctk.CTkButton(orig_zoom_controls, text="+", width=30, command=self.original_zoom_in)
        self.orig_zoom_in_btn.grid(row=0, column=2, padx=5, pady=5)

        self.orig_zoom_label = ctk.CTkLabel(orig_zoom_controls, text="100%")
        self.orig_zoom_label.grid(row=0, column=3, padx=10, pady=5)
        
        # Initially disable zoom buttons
        self.toggle_zoom_controls(False)

    def toggle_zoom_controls(self, enable=True):
        """Enable or disable zoom controls"""
        state = "normal" if enable else "disabled"
        self.orig_zoom_out_btn.configure(state=state)
        self.orig_zoom_reset_btn.configure(state=state)
        self.orig_zoom_in_btn.configure(state=state)
        
        # Only enable compressed zoom controls if there's a compressed image
        comp_state = state if self.compressed_img_np is not None else "disabled"
        self.comp_zoom_out_btn.configure(state=comp_state)
        self.comp_zoom_reset_btn.configure(state=comp_state)
        self.comp_zoom_in_btn.configure(state=comp_state)

    def create_image_display(self):
        # Original image frame
        self.original_frame = ctk.CTkFrame(self.right_frame)
        self.original_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.original_title = ctk.CTkLabel(self.original_frame, text="Original Image", font=ctk.CTkFont(weight="bold"))
        self.original_title.pack(pady=5)
        
        # Create a canvas with scrollbars for original image
        self.original_canvas_frame = ctk.CTkFrame(self.original_frame)
        self.original_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.original_canvas = tk.Canvas(self.original_canvas_frame, bg="#2a2d2e", highlightthickness=0)
        self.original_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.original_scrollbar_y = tk.Scrollbar(self.original_canvas_frame, orient=tk.VERTICAL, command=self.original_canvas.yview)
        self.original_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.original_scrollbar_x = tk.Scrollbar(self.original_frame, orient=tk.HORIZONTAL, command=self.original_canvas.xview)
        self.original_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.original_canvas.configure(xscrollcommand=self.original_scrollbar_x.set, yscrollcommand=self.original_scrollbar_y.set)
        
        # Add mouse wheel binding for zooming original image
        self.original_canvas.bind("<MouseWheel>", self.original_mouse_wheel)  # Windows
        self.original_canvas.bind("<Button-4>", self.original_mouse_wheel)    # Linux scroll up
        self.original_canvas.bind("<Button-5>", self.original_mouse_wheel)    # Linux scroll down
        
        # Compressed image frame
        self.compressed_frame = ctk.CTkFrame(self.right_frame)
        self.compressed_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.compressed_title = ctk.CTkLabel(self.compressed_frame, text="Compressed Image", font=ctk.CTkFont(weight="bold"))
        self.compressed_title.pack(pady=5)
        
        # Create a canvas with scrollbars for compressed image
        self.compressed_canvas_frame = ctk.CTkFrame(self.compressed_frame)
        self.compressed_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.compressed_canvas = tk.Canvas(self.compressed_canvas_frame, bg="#2a2d2e", highlightthickness=0)
        self.compressed_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.compressed_scrollbar_y = tk.Scrollbar(self.compressed_canvas_frame, orient=tk.VERTICAL, command=self.compressed_canvas.yview)
        self.compressed_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.compressed_scrollbar_x = tk.Scrollbar(self.compressed_frame, orient=tk.HORIZONTAL, command=self.compressed_canvas.xview)
        self.compressed_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.compressed_canvas.configure(xscrollcommand=self.compressed_scrollbar_x.set, yscrollcommand=self.compressed_scrollbar_y.set)
        
        # Add mouse wheel binding for zooming compressed image
        self.compressed_canvas.bind("<MouseWheel>", self.compressed_mouse_wheel)  # Windows
        self.compressed_canvas.bind("<Button-4>", self.compressed_mouse_wheel)    # Linux scroll up
        self.compressed_canvas.bind("<Button-5>", self.compressed_mouse_wheel)    # Linux scroll down

    def update_quality_value(self, value):
        """Update the quality value label when slider changes"""
        self.quality_value_label.configure(text=f"{int(value)}")

    def select_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        if file_path:
            self.image_path = file_path
            self.path_label.configure(text=os.path.basename(file_path))
            
            # Load and display the original image
            self.original_image = Image.open(file_path).convert("RGB")
            self.original_img_np = np.array(self.original_image)  # Store numpy version
            
            # Reset zoom factors
            self.original_zoom_factor = 1.0
            self.compressed_zoom_factor = 1.0
            self.orig_zoom_label.configure(text="100%")
            self.comp_zoom_label.configure(text="100%")
            
            # Display the original image
            self.display_original_image()
            
            # Update original image details
            file_size = os.path.getsize(file_path)
            self.original_size = file_size
            self.original_size_label.configure(text=f"Size: {self.format_size(file_size)}")
            self.original_dims_label.configure(text=f"Dimensions: {self.original_image.width} x {self.original_image.height}")
            
            # Enable compress button and zoom controls
            self.compress_button.configure(state="normal")
            self.toggle_zoom_controls(True)
            
            # Reset compression results
            self.compressed_size_label.configure(text="Compressed Size: N/A")
            self.compression_rate_label.configure(text="Compression Rate: N/A")
            self.psnr_label.configure(text="PSNR: N/A")
            
            # Reset compressed image
            self.compressed_canvas.delete("all")
            self.compressed_img_np = None
            
            # Disable compressed zoom controls and save button
            self.comp_zoom_out_btn.configure(state="disabled")
            self.comp_zoom_reset_btn.configure(state="disabled")
            self.comp_zoom_in_btn.configure(state="disabled")
            self.save_button.configure(state="disabled")

    def display_original_image(self):
        """Display the original image in the canvas with zoom"""
        if self.original_image:
            # Clear canvas
            self.original_canvas.delete("all")
            
            # Get the original image dimensions
            img_width, img_height = self.original_image.size
            
            # Apply zoom factor
            new_width = int(img_width * self.original_zoom_factor)
            new_height = int(img_height * self.original_zoom_factor)
            
            # Resize image using PIL
            if self.original_zoom_factor != 1.0:
                resized_img = self.original_image.resize((new_width, new_height), Image.LANCZOS)
            else:
                resized_img = self.original_image
            
            # Convert to PhotoImage and keep reference
            self.tk_original_img = ImageTk.PhotoImage(resized_img)
            
            # Update canvas scroll region
            self.original_canvas.config(scrollregion=(0, 0, new_width, new_height))
            
            # Create image on canvas
            self.original_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_original_img)

    def display_compressed_image(self, compressed_image_np):
        """Display the compressed image in the canvas with zoom"""
        if compressed_image_np is not None:
            # Clear canvas
            self.compressed_canvas.delete("all")
            
            # Store numpy array for later zooming
            self.compressed_img_np = compressed_image_np
            
            # Get dimensions
            img_height, img_width = compressed_image_np.shape[:2]
            
            # Apply zoom factor
            new_width = int(img_width * self.compressed_zoom_factor)
            new_height = int(img_height * self.compressed_zoom_factor)
            
            # Convert to PIL image
            pil_img = Image.fromarray(np.uint8(compressed_image_np))
            
            # Resize image using PIL
            if self.compressed_zoom_factor != 1.0:
                resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            else:
                resized_img = pil_img
            
            # Convert to PhotoImage and keep reference
            self.tk_compressed_img = ImageTk.PhotoImage(resized_img)
            
            # Update canvas scroll region
            self.compressed_canvas.config(scrollregion=(0, 0, new_width, new_height))
            
            # Create image on canvas
            self.compressed_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_compressed_img)
            
            # Enable compressed zoom controls
            self.comp_zoom_out_btn.configure(state="normal")
            self.comp_zoom_reset_btn.configure(state="normal")
            self.comp_zoom_in_btn.configure(state="normal")

    def original_zoom_in(self):
        """Zoom in on the original image"""
        if self.original_image:
            self.original_zoom_factor *= 1.2
            self.update_original_zoom_display()
            self.display_original_image()

    def original_zoom_out(self):
        """Zoom out on the original image"""
        if self.original_image:
            self.original_zoom_factor = max(0.1, self.original_zoom_factor / 1.2)
            self.update_original_zoom_display()
            self.display_original_image()

    def original_zoom_reset(self):
        """Reset zoom on the original image"""
        if self.original_image:
            self.original_zoom_factor = 1.0
            self.update_original_zoom_display()
            self.display_original_image()

    def compressed_zoom_in(self):
        """Zoom in on the compressed image"""
        if self.compressed_img_np is not None:
            self.compressed_zoom_factor *= 1.2
            self.update_compressed_zoom_display()
            self.display_compressed_image(self.compressed_img_np)

    def compressed_zoom_out(self):
        """Zoom out on the compressed image"""
        if self.compressed_img_np is not None:
            self.compressed_zoom_factor = max(0.1, self.compressed_zoom_factor / 1.2)
            self.update_compressed_zoom_display()
            self.display_compressed_image(self.compressed_img_np)

    def compressed_zoom_reset(self):
        """Reset zoom on the compressed image"""
        if self.compressed_img_np is not None:
            self.compressed_zoom_factor = 1.0
            self.update_compressed_zoom_display()
            self.display_compressed_image(self.compressed_img_np)

    def update_original_zoom_display(self):
        """Update the zoom percentage display for original image"""
        zoom_percent = int(self.original_zoom_factor * 100)
        self.orig_zoom_label.configure(text=f"{zoom_percent}%")

    def update_compressed_zoom_display(self):
        """Update the zoom percentage display for compressed image"""
        zoom_percent = int(self.compressed_zoom_factor * 100)
        self.comp_zoom_label.configure(text=f"{zoom_percent}%")

    def original_mouse_wheel(self, event):
        """Handle mouse wheel events for zooming original image"""
        if self.original_image:
            # Check if Control key is pressed for zoom
            if event.state & 0x4:  # Control key
                # Determine zoom direction
                if event.delta > 0 or event.num == 4:  # Zoom in
                    self.original_zoom_in()
                elif event.delta < 0 or event.num == 5:  # Zoom out
                    self.original_zoom_out()
                return "break"  # Prevent default scrolling
            else:
                # Normal scrolling (vertical)
                if event.delta > 0 or event.num == 4:
                    self.original_canvas.yview_scroll(-1, "units")
                elif event.delta < 0 or event.num == 5:
                    self.original_canvas.yview_scroll(1, "units")

    def compressed_mouse_wheel(self, event):
        """Handle mouse wheel events for zooming compressed image"""
        if self.compressed_img_np is not None:
            # Check if Control key is pressed for zoom
            if event.state & 0x4:  # Control key
                # Determine zoom direction
                if event.delta > 0 or event.num == 4:  # Zoom in
                    self.compressed_zoom_in()
                elif event.delta < 0 or event.num == 5:  # Zoom out
                    self.compressed_zoom_out()
                return "break"  # Prevent default scrolling
            else:
                # Normal scrolling (vertical)
                if event.delta > 0 or event.num == 4:
                    self.compressed_canvas.yview_scroll(-1, "units")
                elif event.delta < 0 or event.num == 5:
                    self.compressed_canvas.yview_scroll(1, "units")

    def compress_image(self):
        """Run the JPEG compression process"""
        if self.image_path:
            quality = int(self.quality_slider.get())
            
            # Here we would integrate with your existing compression code
            # For now, I'll create a placeholder that simulates the process
            try:
                # Convert the main function to work with our interface
                # This is where we'll call functions from your existing code
                self.process_image(self.image_path, quality)
                
                # Enable save button after successful compression
                self.save_button.configure(state="normal")
                
            except Exception as e:
                error_window = ctk.CTkToplevel(self)
                error_window.geometry("400x200")
                error_window.title("Error")
                
                error_label = ctk.CTkLabel(
                    error_window, 
                    text=f"An error occurred during compression:\n{str(e)}",
                    wraplength=350
                )
                error_label.pack(padx=20, pady=20)
                
                ok_button = ctk.CTkButton(error_window, text="OK", command=error_window.destroy)
                ok_button.pack(pady=20)

    def process_image(self, image_path, quality):
        """Process the image using the existing JPEG compression functions"""
        from __main__ import (
            Image, np, make_dimensions_divisible_by_block_size, rgb_to_ycbcr, 
            downsample, pad_image_to_block_size, dct_matrix, blockwise_transform, 
            dct_2d_matrix, quantize_dct, get_quantization_tables, zigzag_scan, 
            Run_Length, flatten_rle, process_huffman_encoding, huffman_decode, 
            unflatten_rle, inverse_run_length, inverse_zigzag, dequantize_dct, 
            idct_2d_matrix, unpad_image, ycbcr_to_rgb, psnr
        )
        
        # Load the image
        image = Image.open(image_path).convert("RGB")
        block_size = 8
        
        # Process steps from your main function
        # Ensure image dimensions are divisible by block size
        image_np = np.array(image)
        image_np = make_dimensions_divisible_by_block_size(image_np, block_size)
        
        # Convert RGB to YCrCb
        Y, Cb, Cr = rgb_to_ycbcr(image_np)
        
        # Downsample chrominance channels
        Y_downsampled, Cb_downsampled, Cr_downsampled = downsample(Y, Cb, Cr, "4:2:0")
        
        # Pad images to be divisible by block size
        Y_padded, Y_padding = pad_image_to_block_size(Y_downsampled, block_size)
        Cb_padded, Cb_padding = pad_image_to_block_size(Cb_downsampled, block_size)
        Cr_padded, Cr_padding = pad_image_to_block_size(Cr_downsampled, block_size)
        
        Cb_centered = Cb_padded - 128
        Cr_centered = Cr_padded - 128
        
        # DCT transformation
        D = dct_matrix(block_size)
        Y_dct = blockwise_transform(Y_padded, block_size, dct_2d_matrix, D)
        Cb_dct = blockwise_transform(Cb_centered, block_size, dct_2d_matrix, D)
        Cr_dct = blockwise_transform(Cr_centered, block_size, dct_2d_matrix, D)
        
        # Get quantization tables
        Q_Y, Q_C = get_quantization_tables(quality)
        
        # Quantize the DCT coefficients
        Y_dct_q = quantize_dct(Y_dct, Q_Y)
        Cb_dct_q = quantize_dct(Cb_dct, Q_C)
        Cr_dct_q = quantize_dct(Cr_dct, Q_C)
        
        # Split into blocks and apply zigzag scan
        def split_into_blocks(matrix, block_size=8):
            h, w = matrix.shape
            blocks = []
            for y in range(0, h, block_size):
                for x in range(0, w, block_size):
                    block = matrix[y:y+block_size, x:x+block_size]
                    blocks.append(block)
            return blocks
        
        # For Y channel
        Y_blocks = split_into_blocks(Y_dct_q)
        zigzag_y = []
        for block in Y_blocks:
            zigzag_y.extend(zigzag_scan(block))
            
        # For Cb channel
        Cb_blocks = split_into_blocks(Cb_dct_q)
        zigzag_cb = []
        for block in Cb_blocks:
            zigzag_cb.extend(zigzag_scan(block))
            
        # For Cr channel
        Cr_blocks = split_into_blocks(Cr_dct_q)
        zigzag_cr = []
        for block in Cr_blocks:
            zigzag_cr.extend(zigzag_scan(block))
        
        # Extract DC and AC coefficients
        block_size_z = 64  # 1 DC + 63 AC per block
        num_blocks = len(zigzag_y) // block_size_z
        dc_y = [zigzag_y[i * block_size_z] for i in range(num_blocks)]
        ac_y = [coeff for i in range(num_blocks) for coeff in zigzag_y[i * block_size_z + 1 : (i + 1) * block_size_z]]
        
        num_blocks_cb = len(zigzag_cb) // block_size_z
        dc_cb = [zigzag_cb[i * block_size_z] for i in range(num_blocks_cb)]
        ac_cb = [coeff for i in range(num_blocks_cb) for coeff in zigzag_cb[i * block_size_z + 1 : (i + 1) * block_size_z]]
        
        num_blocks_cr = len(zigzag_cr) // block_size_z
        dc_cr = [zigzag_cr[i * block_size_z] for i in range(num_blocks_cr)]
        ac_cr = [coeff for i in range(num_blocks_cr) for coeff in zigzag_cr[i * block_size_z + 1 : (i + 1) * block_size_z]]
        
        # Apply RLE to AC coefficients
        block_size_ac = 63
        num_blocks_ac = len(ac_y) // block_size_ac
        
        ac_y_blocks = [ac_y[i*block_size_ac : (i+1)*block_size_ac] for i in range(num_blocks_ac)]
        rle_ac_y = []
        for block_ac in ac_y_blocks:
            rle_block = Run_Length(block_ac)
            rle_ac_y.extend(rle_block)
            
        ac_cb_blocks = [ac_cb[i*block_size_ac : (i+1)*block_size_ac] for i in range(num_blocks_ac)]
        rle_ac_cb = []
        for block_ac in ac_cb_blocks:
            rle_block = Run_Length(block_ac)
            rle_ac_cb.extend(rle_block)
            
        ac_cr_blocks = [ac_cr[i*block_size_ac : (i+1)*block_size_ac] for i in range(num_blocks_ac)]
        rle_ac_cr = []
        for block_ac in ac_cr_blocks:
            rle_block = Run_Length(block_ac)
            rle_ac_cr.extend(rle_block)
        
        # Huffman encoding
        symbols_y = flatten_rle(rle_ac_y)
        symbols_cb = flatten_rle(rle_ac_cb)
        symbols_cr = flatten_rle(rle_ac_cr)
        
        res_y = process_huffman_encoding("|".join(symbols_y))
        res_cb = process_huffman_encoding("|".join(symbols_cb))
        res_cr = process_huffman_encoding("|".join(symbols_cr))
        
        # Now decompress for display
        
        # Decode Huffman
        decoded_y = huffman_decode(res_y['coded_message'], res_y['huffman_codes'])
        decoded_cb = huffman_decode(res_cb['coded_message'], res_cb['huffman_codes'])
        decoded_cr = huffman_decode(res_cr['coded_message'], res_cr['huffman_codes'])
        
        # Parse the decoded strings
        rle_y_recovered = unflatten_rle(decoded_y)
        rle_cb_recovered = unflatten_rle(decoded_cb)
        rle_cr_recovered = unflatten_rle(decoded_cr)
        
        # Inverse run-length encoding
        rle_y_decoded = inverse_run_length(rle_y_recovered)
        rle_cb_decoded = inverse_run_length(rle_cb_recovered)
        rle_cr_decoded = inverse_run_length(rle_cr_recovered)
        
        # Organize AC coefficients back into blocks
        ac_blocks_y = [rle_y_decoded[i*63 : (i+1)*63] for i in range(len(rle_y_decoded) // 63)]
        ac_blocks_cb = [rle_cb_decoded[i*63 : (i+1)*63] for i in range(len(rle_cb_decoded) // 63)]
        ac_blocks_cr = [rle_cr_decoded[i*63 : (i+1)*63] for i in range(len(rle_cr_decoded) // 63)]
        
        # Merge DC + AC coefficients for all blocks
        full_zigzag_y = []
        for i in range(len(dc_y)):
            full_zigzag_y.append(dc_y[i])
            full_zigzag_y.extend(ac_blocks_y[i])
            
        full_zigzag_cb = []
        for i in range(len(dc_cb)):
            full_zigzag_cb.append(dc_cb[i])
            full_zigzag_cb.extend(ac_blocks_cb[i])
        
        full_zigzag_cr = []
        for i in range(len(dc_cr)):
            full_zigzag_cr.append(dc_cr[i])
            full_zigzag_cr.extend(ac_blocks_cr[i])
        
        # Inverse zigzag scan
        zigzag_y_decoded = inverse_zigzag(full_zigzag_y)
        zigzag_cb_decoded = inverse_zigzag(full_zigzag_cb)
        zigzag_cr_decoded = inverse_zigzag(full_zigzag_cr)
        
        # Convert blocks to matrices
        h, w = Y_dct_q.shape
        reconstructed_y = np.zeros((h, w), dtype=np.float32)
        block_idx = 0
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                reconstructed_y[y:y+8, x:x+8] = zigzag_y_decoded[block_idx]
                block_idx += 1
                
        h_c, w_c = Cb_dct_q.shape
        reconstructed_cb = np.zeros((h_c, w_c), dtype=np.float32)
        block_idx = 0
        for y in range(0, h_c, 8):
            for x in range(0, w_c, 8):
                reconstructed_cb[y:y+8, x:x+8] = zigzag_cb_decoded[block_idx]
                block_idx += 1
                
        reconstructed_cr = np.zeros((h_c, w_c), dtype=np.float32)
        block_idx = 0
        for y in range(0, h_c, 8):
            for x in range(0, w_c, 8):
                reconstructed_cr[y:y+8, x:x+8] = zigzag_cr_decoded[block_idx]
                block_idx += 1
        
        # Dequantize
        Y_dct_dq = dequantize_dct(reconstructed_y, Q_Y)
        Cb_dct_dq = dequantize_dct(reconstructed_cb, Q_C)
        Cr_dct_dq = dequantize_dct(reconstructed_cr, Q_C)
        
        # Inverse DCT
        Y_idct = blockwise_transform(Y_dct_dq, block_size, idct_2d_matrix, D)
        Cb_idct = blockwise_transform(Cb_dct_dq, block_size, idct_2d_matrix, D)
        Cr_idct = blockwise_transform(Cr_dct_dq, block_size, idct_2d_matrix, D)
        
        # Unpad
        Y_unpadded_center = unpad_image(Y_idct, Y_padding)
        Cb_unpadded_center = unpad_image(Cb_idct, Cb_padding)
        Cr_unpadded_center = unpad_image(Cr_idct, Cr_padding)
        
        # Uncenter chroma
        Cb_unpadded = Cb_unpadded_center + 128
        Cr_unpadded = Cr_unpadded_center + 128
        
        # Clip values
        Y_unpadded = np.clip(Y_unpadded_center, 0, 255)
        Cb_unpadded = np.clip(Cb_unpadded, 0, 255)
        Cr_unpadded = np.clip(Cr_unpadded, 0, 255)
                
        # Convert back to RGB
        reconstructed_image = ycbcr_to_rgb(Y_unpadded, Cb_unpadded, Cr_unpadded)

       # Create temporary file for size calculation
        temp_file = "temp_compressed_image.jpg"
        Image.fromarray(np.uint8(reconstructed_image)).save(temp_file)

        # Store compression metrics
        self.compressed_size = os.path.getsize(temp_file)  # Get actual file size
        self.original_size = os.path.getsize(self.image_path)  # Original file size

        # Calculate compression rate
        self.compression_rate = self.original_size / self.compressed_size if self.compressed_size > 0 else 0

        # Clean up temporary file
        os.remove(temp_file)


        # Calculate metrics
        self.original_size = os.path.getsize(self.image_path)
        self.compression_rate = self.original_size / self.compressed_size if self.compressed_size > 0 else 0

        # Update UI
        self.compressed_size_label.configure(text=f"Compressed Size: {self.format_size(self.compressed_size)}")
        self.compression_rate_label.configure(text=f"Compression Rate: {self.compression_rate:.2f}x")

        # Calculate PSNR
        self.psnr_value = psnr(image_np, reconstructed_image)
        self.psnr_label.configure(text=f"PSNR: {self.psnr_value:.2f} dB")
        
        # Store and display the reconstructed image
        self.compressed_image = reconstructed_image
        self.display_compressed_image(reconstructed_image)

    def save_compressed_image(self):
        """Save the compressed image to disk"""
        if self.compressed_image is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")],
                initialfile=f"Compressed-{os.path.basename(self.image_path)}-q{int(self.quality_slider.get())}",
            )
            
            if file_path:
                # Convert numpy array to PIL Image and save
                Image.fromarray(np.uint8(self.compressed_image)).save(file_path)
                
                # Show confirmation
                confirm_window = ctk.CTkToplevel(self)
                confirm_window.geometry("300x150")
                confirm_window.title("Success")
                
                confirm_label = ctk.CTkLabel(
                    confirm_window, 
                    text=f"Image saved successfully to:\n{os.path.basename(file_path)}"
                )
                confirm_label.pack(padx=20, pady=20)
                
                ok_button = ctk.CTkButton(confirm_window, text="OK", command=confirm_window.destroy)
                ok_button.pack(pady=20)

    def format_size(self, size_in_bytes):
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_in_bytes < 1024.0:
                return f"{size_in_bytes:.2f} {unit}"
            size_in_bytes /= 1024.0
        return f"{size_in_bytes:.2f} TB"

if __name__ == "__main__":
    app = JPEGCompressorApp()
    app.mainloop()