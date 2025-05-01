import heapq
from collections import Counter
import math

# Node class for Huffman Tree
class Node:
    def __init__(self, symbol=None, frequency=None):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.frequency < other.frequency

# Function to calculate character frequencies
def calculate_frequencies(text):
    freq_count = Counter(text)
    total_chars = len(text)
    chars = []
    frequencies = []
    
    for char, count in freq_count.items():
        chars.append(char)
        frequencies.append(count / total_chars)
    
    return chars, frequencies

# Function to calculate entropy
def calculate_entropy(frequencies):
    return -sum(p * math.log2(p) for p in frequencies if p > 0)

# Function to calculate compression metrics
def calculate_compression_metrics(text, huffman_codes, frequencies):
    original_size = len(text) * 8  # ASCII (8 bits per character)
    compressed_size = sum(len(code) * freq * len(text) 
                          for (code), freq in zip(huffman_codes.items(), frequencies))
    table_size = sum(8 + len(code) for code in huffman_codes.values())
    compression_rate = original_size / compressed_size
    compression_rate_with_table = original_size / (compressed_size + table_size)
    avg_code_length = sum(len(code) * freq for (code), freq in zip(huffman_codes.items(), frequencies))
    
    return {
        'avg_code_length': avg_code_length,
        'original_size': original_size,
        'compressed_size': compressed_size,
        'table_size': table_size,
        'compression_rate': compression_rate,
        'compression_rate_with_table': compression_rate_with_table,
        
    }

# Function to build Huffman Tree
def build_huffman_tree(chars, freq):
    priority_queue = [Node(char, f) for char, f in zip(chars, freq)]
    heapq.heapify(priority_queue)
    
    while len(priority_queue) > 1:
        left_child = heapq.heappop(priority_queue)
        right_child = heapq.heappop(priority_queue)
        merged_node = Node(frequency=left_child.frequency + right_child.frequency)
        merged_node.left = left_child
        merged_node.right = right_child
        heapq.heappush(priority_queue, merged_node)
    
    return priority_queue[0]

# Function to generate Huffman codes
def generate_huffman_codes(node, code="", huffman_codes=None):
    if huffman_codes is None:
        huffman_codes = {}
    
    if node is not None:
        if node.symbol is not None:
            huffman_codes[node.symbol] = code
        generate_huffman_codes(node.left, code + "1", huffman_codes)
        generate_huffman_codes(node.right, code + "0", huffman_codes)
    return huffman_codes

# Function to decode Huffman encoded text
def huffman_decode(encoded_text, huffman_codes):
    reverse_mapping = {code: char for char, code in huffman_codes.items()}
    decoded = ""
    current_code = ""
    
    for bit in encoded_text:
        current_code += bit
        if current_code in reverse_mapping:
            decoded += reverse_mapping[current_code]
            current_code = ""
    
    return decoded

# Main function for Huffman encoding
def process_huffman_encoding(text):
    chars, freq = calculate_frequencies(text)
    entropy = calculate_entropy(freq)
    root = build_huffman_tree(chars, freq)
    huffman_codes = generate_huffman_codes(root)
    metrics = calculate_compression_metrics(text, huffman_codes, freq)
    encoded_text = "".join(huffman_codes[char] for char in text)
    decoded_text = huffman_decode(encoded_text, huffman_codes)
    
    return {
        'coded_message': encoded_text,
        'decoded_message': decoded_text,
        'huffman_codes': huffman_codes,
        'entropy': round(entropy, 2),
        **metrics,
        'successful_decode': text == decoded_text
    }