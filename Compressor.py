import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Huffman import process_huffman_encoding,huffman_decode
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

def show_images(original, transformed, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title("Image Originale (RGB)")
    axes[0].axis("off")

    axes[1].imshow(transformed, cmap='gray' if len(transformed.shape) == 2 else None)
    axes[1].set_title(title)
    axes[1].axis("off")

    plt.show()

def show_ycrcb_components(original, Y, Cb, Cr):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original (RGB)")
    axes[0].axis("off")

    axes[1].imshow(Y, cmap='gray')
    axes[1].set_title("Y (Luminance)")
    axes[1].axis("off")

    axes[2].imshow(Cb, cmap='gray')
    axes[2].set_title("Cb (Blue Diff)")
    axes[2].axis("off")

    axes[3].imshow(Cr, cmap='gray')
    axes[3].set_title("Cr (Red Diff)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()
    
    
def showDCT(original, Y , Cb , Cr):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original (RGB)")
    axes[0].axis("off")

    axes[1].imshow(Y, cmap='gray')
    axes[1].set_title("DCT Y")
    axes[1].axis("off")

    axes[2].imshow(Cb, cmap='gray')
    axes[2].set_title("DCT Cb")
    axes[2].axis("off")

    axes[3].imshow(Cr, cmap='gray')
    axes[3].set_title("DCT Cr")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()
    
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

def process_dct_blocks(dct_blocks, block_size, transform_func, D=None):
    """Process 3D array of DCT blocks and combine into a 2D image"""
    
    num_blocks = dct_blocks.shape[0]
    # Calculate dimensions of the output image
    blocks_per_side = int(np.sqrt(num_blocks))  # Assuming square arrangement
    result_height = blocks_per_side * block_size
    result_width = blocks_per_side * block_size
    
    # Initialize output image
    result = np.zeros((result_height, result_width), dtype=np.float32)
    
    # Process each block
    for i in range(num_blocks):
        # Calculate block position
        block_row = (i // blocks_per_side) * block_size
        block_col = (i % blocks_per_side) * block_size
        
        # Apply IDCT transform to this block
        if D is not None:
            transformed_block = transform_func(dct_blocks[i], D)
        else:
            transformed_block = transform_func(dct_blocks[i])
        
        # Place in output image
        result[block_row:block_row+block_size, block_col:block_col+block_size] = transformed_block
    
    return result

def rescale_idct_output(component, original_min, original_max):
    """Rescale IDCT output to match expected YCbCr ranges"""
    # For Y, we want 0-255
    # For Cb/Cr, we want 0-255 (centered at 128)
    if original_max == original_min:
        return np.ones_like(component) * original_min
    
    # Scale to original range
    return (component - np.min(component)) / (np.max(component) - np.min(component)) * \
           (original_max - original_min) + original_min


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

def show_quantized(dct_q_y, dct_q_cb, dct_q_cr):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.log(1 + np.abs(dct_q_y)), cmap='gray')
    axes[0].set_title("Quantized Y DCT Coefficients")
    axes[0].axis("off")

    axes[1].imshow(np.log(1 + np.abs(dct_q_cb)), cmap='gray')
    axes[1].set_title("Quantized Cb DCT Coefficients")
    axes[1].axis("off")

    axes[2].imshow(np.log(1 + np.abs(dct_q_cr)), cmap='gray')
    axes[2].set_title("Quantized Cr DCT Coefficients")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

def show_idct(Y, Cb, Cr):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(Y, cmap='gray')
    axes[0].set_title("IDCT Y")
    axes[0].axis("off")

    axes[1].imshow(Cb, cmap='gray')
    axes[1].set_title("IDCT Cb")
    axes[1].axis("off")

    axes[2].imshow(Cr, cmap='gray')
    axes[2].set_title("IDCT Cr")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

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
            for i in range(max(0, s - cols + 1), min(rows, s + 1)):
                result.append(matrix[i][s - i])
        else:
            for i in range(max(0, s - rows + 1), min(cols, s + 1)):
                result.append(matrix[s - i][i])
    return result

def Run_Length(data):
    # Assume first value is DC, skip it
    ac_coeffs = data[1:]
    rle = []
    zero_count = 0

    for coeff in ac_coeffs:
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


def inverse_zigzag(zigzag_data, rows=8, cols=8):
    """
    Convert 1D zigzag-scanned data back to 2D matrix
    
    Args:
        zigzag_data: 1D list of zigzag-scanned data
        rows: Number of rows in the output matrix
        cols: Number of columns in the output matrix
        
    Returns:
        2D matrix with data arranged according to inverse zigzag scan
    """
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    idx = 0
    
    for s in range(rows + cols - 1):
        if s % 2 == 0:  # Even sum: go up
            for i in range(max(0, s - cols + 1), min(rows, s + 1)):
                if idx < len(zigzag_data):
                    result[i][s - i] = zigzag_data[idx]
                    idx += 1
        else:  # Odd sum: go down
            for i in range(max(0, s - rows + 1), min(cols, s + 1)):
                if idx < len(zigzag_data):
                    result[s - i][i] = zigzag_data[idx]
                    idx += 1
    
    return result

def flatten_rle(rle):
    return [f"{run}/{val}" for run, val in rle]

def unflatten_rle(decoded_string):
    """
    Convert a Huffman-decoded string back to RLE format (list of tuples)
    With additional debugging and error handling
    
    Args:
        decoded_string: String from Huffman decoding
        
    Returns:
        List of (zero_count, value) tuples
    """
    # Debug information
    print("Type of decoded string:", type(decoded_string))
    print("Length of decoded string:", len(decoded_string))
    print("First 20 characters:", decoded_string[:20])
    
    # Create a safe version with better error handling
    rle_data = []
    
    # For now, use a simple approach - every two characters represent a tuple
    i = 0
    while i < len(decoded_string) - 1:  # Ensure we have at least 2 chars
        try:
            # Try to extract two consecutive characters as run and value
            run = int(decoded_string[i]) if decoded_string[i].isdigit() else 0
            val = int(decoded_string[i+1]) if decoded_string[i+1].isdigit() else 0
            rle_data.append((run, val))
        except (ValueError, IndexError):
            # If conversion fails, skip this pair
            pass
        i += 2
    
    # If we couldn't parse anything, add at least one EOB marker
    if not rle_data:
        rle_data.append((0, 0))  # EOB marker
    
    print("Parsed RLE data (first 5 tuples):", rle_data[:5])
    return rle_data

def inverse_run_length(rle_data, dc_value=0):
    """
    Convert run-length encoded data back to zigzag format
    
    Args:
        rle_data: A list of (zero_count, value) tuples from run-length encoding
        dc_value: The DC coefficient value
        
    Returns:
        A list containing the decoded values in zigzag order
    """
    # Start with the DC coefficient
    decoded = [dc_value]
    
    # Process AC coefficients
    for zero_count, value in rle_data:
        # EOB marker
        if zero_count == 0 and value == 0:
            break
            
        # Add zeros
        decoded.extend([0] * zero_count)
        
        # Add the non-zero value
        decoded.append(value)
    
    # Ensure we have a full block (8x8 = 64 values)
    while len(decoded) < 64:
        decoded.append(0)
        
    return decoded

# Alternative temporary solution if parsing fails completely
def safe_zigzag_data():
    """
    Create safe zigzag data as a fallback
    Returns a list of 64 zeros (8x8 block)
    """
    return [0] * 64

def main():
    image_path = 'ford-mustang-bullitt-1920x774px.jpg'
    image = Image.open(image_path).convert("RGB")
    block_size = 8
    
    # Ensure image dimensions are divisible by block size
    image = make_dimensions_divisible_by_block_size(np.array(image), block_size)
    
    # Convert RGB to YCrCb
    Y, Cb, Cr = rgb_to_ycbcr(image)
    print("Converted to YCrCb")
    print("Y range:", Y.min(), Y.max())
    print("Cb range:", Cb.min(), Cb.max())
    print("Cr range:", Cr.min(), Cr.max())
    
    # Downsample chrominance channels
    Y_downsampled, Cb_downsampled, Cr_downsampled = downsample(Y, Cb, Cr, "4:2:0")
    print("Downsampled YCrCb")
    
    # Show original and YCrCb components
    show_ycrcb_components(image, Y_downsampled, Cb_downsampled, Cr_downsampled)
    
    # Pad images to be divisible by block size
    Y_padded, Y_padding = pad_image_to_block_size(Y_downsampled, block_size)
    Cb_padded, Cb_padding = pad_image_to_block_size(Cb_downsampled, block_size)
    Cr_padded, Cr_padding = pad_image_to_block_size(Cr_downsampled, block_size)
    
    # DCT transformation
    D = dct_matrix(block_size)
    Y_dct = blockwise_transform(Y_padded, block_size, dct_2d_matrix, D)
    Cb_dct = blockwise_transform(Cb_padded, block_size, dct_2d_matrix, D)
    Cr_dct = blockwise_transform(Cr_padded, block_size, dct_2d_matrix, D)
    
    print(f"Original Y range: {Y.min()} to {Y.max()}")
    print(f"Y DCT range before quantization: {Y_dct.min()} to {Y_dct.max()}")
    
    # Display DCT coefficients
    Y_dct_log = log_scale_dct(Y_dct)
    Cb_dct_log = log_scale_dct(Cb_dct)
    Cr_dct_log = log_scale_dct(Cr_dct)
    print("DCT transformation completed")
    showDCT(image, Y_dct_log, Cb_dct_log, Cr_dct_log)
    
    # Get quality factor and quantization tables
    quality = int(input("Enter quality factor (0-50, where 0=best, 50=worst): "))
    quality = max(0, min(50, quality))
    Q_Y, Q_C = get_quantization_tables(quality)
    print(f"Q_Y min/max: {Q_Y.min()} to {Q_Y.max()}")
    
    # Quantize the DCT coefficients
    Y_dct_q = quantize_dct(Y_dct, Q_Y)
    Cb_dct_q = quantize_dct(Cb_dct, Q_C)
    Cr_dct_q = quantize_dct(Cr_dct, Q_C)
    
    print(f"Y DCT quantized range: {Y_dct_q.min()} to {Y_dct_q.max()}")
    
    print("Quantization completed")
    
    zigzag_y = zigzag_scan(Y_dct_q)
    zigzag_cb = zigzag_scan(Cb_dct_q)
    zigzag_cr = zigzag_scan(Cr_dct_q)
    print("Zigzag scan completed")
    
    rle_y = Run_Length(zigzag_y)
    rle_cb = Run_Length(zigzag_cb)
    rle_cr = Run_Length(zigzag_cr)
    print("Run-length encoding completed")
    
    # Huffman encoding
    symbols_y = flatten_rle(rle_y)
    symbols_cb = flatten_rle(rle_cb)
    symbols_cr = flatten_rle(rle_cr)

    print("Performing Huffman encoding...")

    res_y = process_huffman_encoding("".join(symbols_y))
    res_cb = process_huffman_encoding("".join(symbols_cb))
    res_cr = process_huffman_encoding("".join(symbols_cr))

    # Decompressing the data
    print("Decompressing the data...")
    # Decode the Huffman encoded data
    decoded_y = huffman_decode(res_y['coded_message'], res_y['huffman_codes'])
    decoded_cb = huffman_decode(res_cb['coded_message'], res_cb['huffman_codes'])
    decoded_cr = huffman_decode(res_cr['coded_message'], res_cr['huffman_codes'])
            
    
    try:
        # Try to parse the decoded strings
        rle_y_recovered = unflatten_rle(decoded_y)
        rle_cb_recovered = unflatten_rle(decoded_cb)
        rle_cr_recovered = unflatten_rle(decoded_cr)
        
        # Inverse run-length encoding
        rle_y_decoded = inverse_run_length(rle_y_recovered)
        rle_cb_decoded = inverse_run_length(rle_cb_recovered)
        rle_cr_decoded = inverse_run_length(rle_cr_recovered)
    except Exception as e:
        # If parsing fails, use safe fallback
        print(f"Error during RLE processing: {e}")
        print("Using safe fallback data")
        rle_y_decoded = safe_zigzag_data()
        rle_cb_decoded = safe_zigzag_data()
        rle_cr_decoded = safe_zigzag_data()
        
        
    # Inverse zigzag scan
    zigzag_y_decoded = inverse_zigzag(rle_y_decoded)
    zigzag_cb_decoded = inverse_zigzag(rle_cb_decoded)
    zigzag_cr_decoded = inverse_zigzag(rle_cr_decoded)
    
    
    # Reshape back to 8x8 blocks
    combined_y_matrix = np.array(zigzag_y_decoded).reshape(-1, 64).reshape(-1, 8, 8)
    Y_dct_dq = np.array([dequantize_dct(block, Q_Y) for block in combined_y_matrix])
    print(f"Y DCT dequantized range: {Y_dct_dq.min()} to {Y_dct_dq.max()}")
    
    combined_cb_matrix = np.array(zigzag_cb_decoded).reshape(-1, 64).reshape(-1, 8, 8)
    Cb_dct_dq = np.array([dequantize_dct(block, Q_C) for block in combined_cb_matrix])
    
    combined_cr_matrix = np.array(zigzag_cr_decoded).reshape(-1, 64).reshape(-1, 8, 8)
    Cr_dct_dq = np.array([dequantize_dct(block, Q_C) for block in combined_cr_matrix])
    
    
            
    
    # Display quantized DCT coefficients
    show_quantized(Y_dct_q, Cb_dct_q, Cr_dct_q)
    

    print("Dequantization completed")
    
    # Inverse 
    # After defining your process_dct_blocks function:

    print(f"Y_dct_dq shape and sample: {Y_dct_dq.shape}, {Y_dct_dq[0, 0]}")
    
    # If Y is all zeros, use Cb or Cr data instead
    if Y_dct_dq.max() == 0:
        print("WARNING: Y channel is all zeros - using artificial luminance!")
        # Create artificial Y data from Cb/Cr or set to mid-gray
        Y_idct = np.ones((block_size, block_size)) * 128  # Mid-gray
    else:
        Y_idct = process_dct_blocks(Y_dct_dq, block_size, idct_2d_matrix, D)
    
    Cb_idct = process_dct_blocks(Cb_dct_dq, block_size, idct_2d_matrix, D)
    Cr_idct = process_dct_blocks(Cr_dct_dq, block_size, idct_2d_matrix, D)

    
    # Remove padding
    Y_unpadded = unpad_image(Y_idct, Y_padding)
    Cb_unpadded = unpad_image(Cb_idct, Cb_padding)
    Cr_unpadded = unpad_image(Cr_idct, Cr_padding)
    
    print(f"Y range1: {Y_unpadded.min()} to {Y_unpadded.max()}")
    print(f"Cb range: {Cb_unpadded.min()} to {Cb_unpadded.max()}")
    print(f"Cr range: {Cr_unpadded.min()} to {Cr_unpadded.max()}")
    
    # Clip values to valid range
    Y_unpadded = np.clip(Y_unpadded, 0, 255)
    Cb_unpadded = np.clip(Cb_unpadded, 0, 255)
    Cr_unpadded = np.clip(Cr_unpadded, 0, 255)
    
    print(f"Y range2: {Y_unpadded.min()} to {Y_unpadded.max()}")
    print(f"Cb range: {Cb_unpadded.min()} to {Cb_unpadded.max()}")
    print(f"Cr range: {Cr_unpadded.min()} to {Cr_unpadded.max()}")
    
    Y_unpadded = np.repeat(np.repeat(Y_unpadded, 2, axis=0), 2, axis=1)
    Cb_unpadded = np.repeat(np.repeat(Cb_unpadded, 2, axis=0), 2, axis=1)
    Cr_unpadded = np.repeat(np.repeat(Cr_unpadded, 2, axis=0), 2, axis=1)
    
    # Show the IDCT components
    show_idct(Y_unpadded, Cb_unpadded, Cr_unpadded)
    
    print("Y range after IDCT:", Y_unpadded.min(), Y_unpadded.max())
    print("Cb range after IDCT:", Cb_unpadded.min(), Cb_unpadded.max())
    print("Cr range after IDCT:", Cr_unpadded.min(), Cr_unpadded.max())
    
    print(f"Shapes before conversion: Y={Y_unpadded.shape}, Cb={Cb_unpadded.shape}, Cr={Cr_unpadded.shape}")
    
    # Convert back to RGB
    # Before color conversion, artificially set Y to a mid-range value
    reconstructed_image = ycbcr_to_rgb(Y_unpadded, Cb_unpadded, Cr_unpadded)
    
    # Show the reconstructed image
    plt.figure(figsize=(10, 8))
    plt.imshow(reconstructed_image)
    plt.title("Reconstructed Image")
    plt.axis("off")
    plt.show()
    
    # Calculate PSNR between original and reconstructed images
    psnr_value = psnr(image, reconstructed_image)
    print(f"PSNR: {psnr_value:.2f} dB")
    
if __name__ == "__main__":
    main()