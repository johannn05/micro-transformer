# quantize_utils.py
import torch
import torch.nn as nn
import time
from pathlib import Path

def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def apply_dynamic_quantization(model):
    """
    Apply dynamic quantization to the model's linear layers
    """
    # Make sure model is in eval mode
    model.eval()
    
    # Apply dynamic quantization to nn.Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,                 # the original model
        {torch.nn.Linear},     # layers to dynamically quantize
        dtype=torch.qint8      # target dtype for quantized weights
    )
    
    return quantized_model

def measure_inference_time(model, data, device, num_runs=50):
    """Measure average inference time"""
    model.eval()
    encoder_input = data["encoder_input"].to(device)
    encoder_mask = data["encoder_mask"].to(device)
    decoder_input = data["decoder_input"].to(device) 
    decoder_mask = data["decoder_mask"].to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            _ = model.project(decoder_output)
    
    # Measure time
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            _ = model.project(decoder_output)
    end = time.time()
    
    return (end - start) / num_runs

def compare_models(original_model, quantized_model, sample_data, device):
    """
    Compare performance metrics between original and quantized models
    """
    # Size comparison
    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)
    
    # Speed comparison
    original_time = measure_inference_time(original_model, sample_data, device)
    quantized_time = measure_inference_time(quantized_model, sample_data, device)
    
    results = {
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "size_reduction_percent": (1 - quantized_size/original_size) * 100,
        "original_inference_ms": original_time * 1000,
        "quantized_inference_ms": quantized_time * 1000,
        "speedup_factor": original_time / quantized_time
    }
    
    return results

def print_comparison_table(results):
    """Print a formatted comparison table with careful type handling"""
    print("\n" + "="*60)
    print("MODEL COMPARISON: ORIGINAL vs QUANTIZED")
    print("="*60)
    print(f"{'Metric':<30} | {'Original':<12} | {'Quantized':<12} | {'Change':<12}")
    print("-"*60)
    
    # Ensure values are converted to floats before formatting
    try:
        orig_size = float(results['original_size_mb'])
        quant_size = float(results['quantized_size_mb'])
        reduction = float(results['size_reduction_percent'])
        orig_time = float(results['original_inference_ms'])
        quant_time = float(results['quantized_inference_ms'])
        speedup = float(results['speedup_factor'])
        
        print(f"{'Model Size (MB)':<30} | {orig_size:<12.2f} | {quant_size:<12.2f} | {'-' + str(reduction):<11.2f}%")
        print(f"{'Inference Time (ms)':<30} | {orig_time:<12.2f} | {quant_time:<12.2f} | {speedup:<11.2f}x")
    except (ValueError, TypeError) as e:
        # Fallback for debugging - show raw values
        print(f"Error formatting values: {e}")
        print(f"Raw results: {results}")
        # Simple fallback formatting without the .2f
        print(f"{'Model Size (MB)':<30} | {results['original_size_mb']} | {results['quantized_size_mb']} | -{results['size_reduction_percent']}%")
        print(f"{'Inference Time (ms)':<30} | {results['original_inference_ms']} | {results['quantized_inference_ms']} | {results['speedup_factor']}x")
    
    print("="*60)