import torch
"""
This script quantizes the weights of a PyTorch model and saves the quantized weights.
Functions:
    quantize_tensor(tensor, num_bits=4):
        Quantizes a given tensor to a specified number of bits.
        Args:
            tensor (torch.Tensor): The tensor to be quantized.
            num_bits (int): The number of bits to use for quantization. Default is 4.
        Returns:
            torch.Tensor: The dequantized tensor.
Main Process:
    1. Load model weights from a file.
    2. Create a new dictionary to store quantized weights.
    3. Iterate through all weights and quantize them if they are weight tensors.
    4. Save the quantized model weights to a file.
"""
def quantize_tensor(tensor, num_bits=4):
    qmin = -2**(num_bits - 1)
    qmax = 2**(num_bits - 1) - 1

    # Flatten and order the tensor
    sorted_tensor = tensor.flatten().sort().values
    n = sorted_tensor.numel()
    
    # Determination of the 5% and 95% positions
    lower_index = int(n * 0.01)
    upper_index = int(n * 0.99)
    
    lower_bound = sorted_tensor[lower_index].item()
    upper_bound = sorted_tensor[upper_index].item()
    
    # Calculate the scale and zero_point of the center section.
    scale = (upper_bound - lower_bound) / (qmax - qmin)
    zero_point = qmin - lower_bound / scale
    
    # quantization
    q_tensor = tensor.clone().float()
    q_tensor = (q_tensor / scale + zero_point).round().clamp(qmin, qmax)

    # Special Handling Minimum 5% and Maximum 5%
    q_tensor[tensor <= lower_bound] = qmin
    q_tensor[tensor >= upper_bound] = qmax

    q_tensor = q_tensor.to(torch.int8)  # Make sure the tensor is in int8 format
    
    # Edge quantization
    dequantized_tensor = scale * (q_tensor.float() - zero_point)
    dequantized_tensor[q_tensor == qmin] = lower_bound  # Invert the smallest 5% to lower_bound
    dequantized_tensor[q_tensor == qmax] = upper_bound  # Invert the biggest 5% to upper_bound
    print(lower_bound)
    
    return dequantized_tensor


# Loading model weights
model_weights = torch.load("spiking_model_weights.pth")

# Create a new dictionary to store the quantized weights
quantized_model_weights = {}

# Iterate over all weights and quantize
for key, value in model_weights.items():
    if "weight" in key:
        q_weight = quantize_tensor(value)
        quantized_model_weights[key] = q_weight
        #quantized_model_weights[key + "_scale"] = scale
        #quantized_model_weights[key + "_zero_point"] = zero_point
    else:
        quantized_model_weights[key] = value

# Save quantized model weights
torch.save(quantized_model_weights, "spiking_model_quantized_weights.pth")
