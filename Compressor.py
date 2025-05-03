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
def process_dct_blocks(dct_blocks, block_size, transform_func, D=None, original_shape=None,img=None):
    """
    Process DCT blocks and combine into a 2D image
    """
    if original_shape:
        result_height, result_width = original_shape
    else:
        print("WARNING: No original shape provided for process_dct_blocks")
        return None
    
    # Calculate required blocks for the image
    blocks_height = (result_height + block_size - 1) // block_size
    blocks_width = (result_width + block_size - 1) // block_size
    total_blocks_needed = blocks_height * blocks_width
    
    print(f"Need {blocks_height}x{blocks_width}={total_blocks_needed} blocks for shape {original_shape} of {img}")
    
    # Check if we have enough blocks
    if len(dct_blocks) < total_blocks_needed:
        print(f"WARNING: Not enough blocks! Have {len(dct_blocks)}, need {total_blocks_needed} of {img}")
        # Duplicate the existing blocks to fill the image (better than black)
        expanded_blocks = []
        for i in range(total_blocks_needed):
            expanded_blocks.append(dct_blocks[i % len(dct_blocks)])
        dct_blocks = np.array(expanded_blocks)
        print(f"Expanded blocks to {len(dct_blocks)} of {img}")
    
    # Initialize output image
    result = np.zeros((result_height, result_width), dtype=np.float32)
    
    # Process each block
    block_index = 0
    for i in range(blocks_height):
        for j in range(blocks_width):
            if block_index >= len(dct_blocks):
                break
                
            block_row = i * block_size
            block_col = j * block_size
            
            # Handle potential partial blocks at edges
            h = min(block_size, result_height - block_row)
            w = min(block_size, result_width - block_col)
            
            # Apply transform to the block
            if D is not None:
                transformed_block = transform_func(dct_blocks[block_index], D)
            else:
                transformed_block = transform_func(dct_blocks[block_index])
            
            # Place in output image
            result[block_row:block_row+h, block_col:block_col+w] = transformed_block[:h, :w]
            block_index += 1
    
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
    # show_ycrcb_components(image, Y_downsampled, Cb_downsampled, Cr_downsampled)
    
    # Pad images to be divisible by block size
    Y_padded, Y_padding = pad_image_to_block_size(Y_downsampled, block_size)
    Cb_padded, Cb_padding = pad_image_to_block_size(Cb_downsampled, block_size)
    Cr_padded, Cr_padding = pad_image_to_block_size(Cr_downsampled, block_size)
    
    Cb_centered = Cb_padded - 128
    Cr_centered = Cr_padded - 128
    print("Cb_centered range:", Cb_centered.min(), Cb_centered.max())
    print("Cr_centered range:", Cr_centered.min(), Cr_centered.max())
    # DCT transformation
    D = dct_matrix(block_size)
    Y_dct = blockwise_transform(Y_padded, block_size, dct_2d_matrix, D)
    Cb_dct = blockwise_transform(Cb_centered, block_size, dct_2d_matrix, D)
    Cr_dct = blockwise_transform(Cr_centered, block_size, dct_2d_matrix, D)
    
    print(f"Original Y range: {Y.min()} to {Y.max()}")
    print(f"Y DCT range before quantization: {Y_dct.min()} to {Y_dct.max()}")
    
    # Display DCT coefficients
    Y_dct_log = log_scale_dct(Y_dct)
    Cb_dct_log = log_scale_dct(Cb_dct)
    Cr_dct_log = log_scale_dct(Cr_dct)
    print("DCT transformation completed")
    # showDCT(image, Y_dct_log, Cb_dct_log, Cr_dct_log)
    
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
    
    # Display quantized DCT coefficients
    # show_quantized(Y_dct_q, Cb_dct_q, Cr_dct_q)
    
    print("Quantization completed")
    
    print("before zigzag Y:", Y_dct_q[:10])
    print("before zigzag Cb:", Cb_dct_q[:10])
    print("before zigzag Cr:", Cr_dct_q[:10])
    
    zigzag_y = zigzag_scan(Y_dct_q)
    zigzag_cb = zigzag_scan(Cb_dct_q)
    zigzag_cr = zigzag_scan(Cr_dct_q)
    
    print("Zigzag scan Y:", zigzag_y[:10])
    print("Zigzag scan Cb:", zigzag_cb[:10])
    print("Zigzag scan Cr:", zigzag_cr[:10])

    print("Zigzag scan completed")
    
    # For Y channel:
    block_size_z = 64  # 1 DC + 63 AC per block
    num_blocks = len(zigzag_y) // block_size_z
    dc_y = [zigzag_y[i * block_size_z] for i in range(num_blocks)]  # All DC coefficients
    ac_y = [coeff for i in range(num_blocks) for coeff in zigzag_y[i * block_size_z + 1 : (i + 1) * block_size_z]]  # All AC
    
    # For Cb and Cr channels:
    num_blocks_cb = len(zigzag_cb) // block_size_z
    dc_cb = [zigzag_cb[i * block_size_z] for i in range(num_blocks_cb)]  # All DC coefficients
    ac_cb = [coeff for i in range(num_blocks_cb) for coeff in zigzag_cb[i * block_size_z + 1 : (i + 1) * block_size_z]]  # All AC
    
    num_blocks_cr = len(zigzag_cr) // block_size_z
    dc_cr = [zigzag_cr[i * block_size_z] for i in range(num_blocks_cr)]  # All DC coefficients
    ac_cr = [coeff for i in range(num_blocks_cr) for coeff in zigzag_cr[i * block_size_z + 1 : (i + 1) * block_size_z]]  # All AC
    
    print("Number of DC_Y:", len(dc_y))  # Should be 23040
    print("Number of DC_Cb:", len(dc_cb))  # Should be 5760
    print("Number of DC_Cr:", len(dc_cr))  # Should be 5760
    
    print("AC Y:", ac_y[:10])
    print("AC Cb:", ac_cb[:10])
    print("AC Cr:", ac_cr[:10])



    # Split AC coefficients into per-block chunks
    block_size_ac = 63
    num_blocks_ac = len(ac_y) // block_size_ac
    ac_y_blocks = [ac_y[i*block_size_ac : (i+1)*block_size_ac] for i in range(num_blocks_ac)]

    # Apply RLE to each block's AC coefficients
    rle_ac_y = []
    for block_ac in ac_y_blocks:
        rle_block = Run_Length(block_ac)  # EOB added per block
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
    print('rle_ac_y:', rle_ac_y[:10])
    print('rle_ac_cb:', rle_ac_cb[:10])
    print('rle_ac_cr:', rle_ac_cr[:10])


    print("Run-length encoding completed")
    
    # Huffman encoding
    symbols_y = flatten_rle(rle_ac_y)
    symbols_cb = flatten_rle(rle_ac_cb)
    symbols_cr = flatten_rle(rle_ac_cr)#correct 
    
    print("Symbols Y:", symbols_y[:10])
    print("Symbols Cb:", symbols_cb[:10])
    print("Symbols Cr:", symbols_cr[:10])

    print("Performing Huffman encoding...")

    res_y = process_huffman_encoding("|".join(symbols_y))
    res_cb = process_huffman_encoding("|".join(symbols_cb))
    res_cr = process_huffman_encoding("|".join(symbols_cr))#correct 


    # Decompressing the data
    print("Decompressing the data...")
    # Decode the Huffman encoded data
    decoded_y = huffman_decode(res_y['coded_message'], res_y['huffman_codes'])
    decoded_cb = huffman_decode(res_cb['coded_message'], res_cb['huffman_codes'])
    decoded_cr = huffman_decode(res_cr['coded_message'], res_cr['huffman_codes'])#correct 
    
    # print("Decoded Y:", decoded_y[:10])
    # print("Decoded Cb:", decoded_cb[:10])
    # print("Decoded Cr:", decoded_cr[:10])
            
    
    try:
        # Try to parse the decoded strings
        rle_y_recovered = unflatten_rle(decoded_y)
        rle_cb_recovered = unflatten_rle(decoded_cb)
        rle_cr_recovered = unflatten_rle(decoded_cr)#correct 
        
        print("after unflattening", rle_y_recovered[:10])
        print("after unflattening", rle_cb_recovered[:10])
        print("after unflattening", rle_cr_recovered[:10])
        
        # Inverse run-length encoding
        rle_y_decoded = inverse_run_length(rle_y_recovered)
        rle_cb_decoded = inverse_run_length(rle_cb_recovered)
        rle_cr_decoded = inverse_run_length(rle_cr_recovered) ##correct 
        
        print("after inverse run-length Y", rle_y_decoded[:10])
        print("after inverse run-length Cb", rle_cb_decoded[:10])
        print("after inverse run-length Cr", rle_cr_decoded[:10])
        
        ac_blocks_y = [rle_y_decoded[i*63 : (i+1)*63] for i in range(len(rle_y_decoded) // 63)]
        ac_blocks_cb = [rle_cb_decoded[i*63 : (i+1)*63] for i in range(len(rle_cb_decoded) // 63)]
        ac_blocks_cr = [rle_cr_decoded[i*63 : (i+1)*63] for i in range(len(rle_cr_decoded) // 63)]

        
            # Merge DC + AC coefficients for all blocks
        full_zigzag_y = []
        for i in range(len(dc_y)):  # dc_y should have 1 entry per block
            full_zigzag_y.append(dc_y[i])  # Add DC
            full_zigzag_y.extend(ac_blocks_y[i]) # AC

        full_zigzag_cb = []
        for i in range(len(dc_cb)):
            full_zigzag_cb.append(dc_cb[i])
            full_zigzag_cb.extend(ac_blocks_cb[i])

        full_zigzag_cr = []
        for i in range(len(dc_cr)):
            full_zigzag_cr.append(dc_cr[i])
            full_zigzag_cr.extend(ac_blocks_cr[i])

        print("full_zigzag_y:", full_zigzag_y[:10])
        print("full_zigzag_cb:", full_zigzag_cb[:10])
        print("full_zigzag_cr:", full_zigzag_cr[:10])

    except Exception as e:
        # If parsing fails, use safe fallback
        print(f"Error during RLE processing: {e}")
        print("Using safe fallback data")
        rle_y_decoded = safe_zigzag_data()
        rle_cb_decoded = safe_zigzag_data()
        rle_cr_decoded = safe_zigzag_data()
        
        
    # Inverse zigzag scan
    zigzag_y_decoded = inverse_zigzag(full_zigzag_y)
    zigzag_cb_decoded = inverse_zigzag(full_zigzag_cb)
    zigzag_cr_decoded = inverse_zigzag(full_zigzag_cr)
    
    # print("inverse zigzag Y:", zigzag_y_decoded[:10])
    # print("inverse zigzag Cb:", zigzag_cb_decoded[:10])
    # print("inverse zigzag Cr:", zigzag_cr_decoded[:10])
    
    # ====== Convert 3D blocks to 2D matrices ======
    # For Y channel
    h, w = Y_dct_q.shape
    reconstructed_y = np.zeros((h, w), dtype=np.float32)
    block_idx = 0
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            reconstructed_y[y:y+8, x:x+8] = zigzag_y_decoded[block_idx]
            block_idx += 1

    # For Cb/Cr (adjust dimensions accordingly)
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

    
    # After inverse zigzag
    print(f"First zigzag_y_decoded block shape: {np.array(zigzag_y_decoded[:64]).shape}")
    
    print("dct_y_decoded:", reconstructed_y[:10])
    print("Y_dct_q:", Y_dct_q[:10])
    
    print("dct_cb_decoded:", reconstructed_cb[:10])
    print("Cb_dct_q:", Cb_dct_q[:10])
    
    print("dct_cr_decoded:", reconstructed_cr[:10])
    print("Cr_dct_q:", Cr_dct_q[:10])

    Y_dct_dq = dequantize_dct(reconstructed_y, Q_Y)
    Cb_dct_dq = dequantize_dct(reconstructed_cb, Q_C)
    Cr_dct_dq = dequantize_dct(reconstructed_cr, Q_C)
    
    print("Dequantization completed")
    
    # Inverse 

    print(f"Y_dct_dq shape and sample: {Y_dct_dq.shape}, {Y_dct_dq[0, 0]}")

    # Store original shapes before padding
    Y_original_shape = Y_downsampled.shape
    Cb_original_shape = Cb_downsampled.shape
    Cr_original_shape = Cr_downsampled.shape

    # Inverse DCT
    Y_idct = blockwise_transform(Y_dct_dq, block_size, idct_2d_matrix, D)
    Cb_idct = blockwise_transform(Cb_dct_dq, block_size, idct_2d_matrix, D)
    Cr_idct = blockwise_transform(Cr_dct_dq, block_size, idct_2d_matrix, D)
    
    # 2) Unpad the *centered* IDCT outputs
    Y_unpadded_center  = unpad_image(Y_idct,  Y_padding)
    Cb_unpadded_center = unpad_image(Cb_idct, Cb_padding)
    Cr_unpadded_center = unpad_image(Cr_idct, Cr_padding)
    
    # 3) Now “uncenter” the chroma by adding +128
    Cb_unpadded = Cb_unpadded_center + 128
    Cr_unpadded = Cr_unpadded_center + 128
    
    # 4) Clip into [0,255]
    Y_unpadded  = np.clip(Y_unpadded_center, 0, 255)
    Cb_unpadded = np.clip(Cb_unpadded,       0, 255)
    Cr_unpadded = np.clip(Cr_unpadded,       0, 255)

    
    print(f"Y range2: {Y_unpadded.min()} to {Y_unpadded.max()}")
    print(f"Cb range: {Cb_unpadded.min()} to {Cb_unpadded.max()}")
    print(f"Cr range: {Cr_unpadded.min()} to {Cr_unpadded.max()}")

    
    # Show the IDCT components
    show_idct(Y_unpadded, Cb_unpadded, Cr_unpadded)
    
    print("Y range after IDCT:", Y_unpadded.min(), Y_unpadded.max())
    print("Cb range after IDCT:", Cb_unpadded.min(), Cb_unpadded.max())
    print("Cr range after IDCT:", Cr_unpadded.min(), Cr_unpadded.max())
    
    print(f"Shapes before conversion: Y={Y_unpadded.shape}, Cb={Cb_unpadded.shape}, Cr={Cr_unpadded.shape}")
    
    # Convert back to RGB
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