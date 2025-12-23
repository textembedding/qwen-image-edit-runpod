# Use a modern Runpod PyTorch base image
FROM runpod/pytorch:1.0.3-cu1290-torch290-ubuntu2204

# Install dependencies
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors pillow runpod hf_transfer bitsandbytes git+https://github.com/huggingface/diffusers

# (Optional) Pre-download the model to reduce cold start latency
RUN python -c 'import torch; from diffusers import QwenImageEditPlusPipeline; QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)'

# Copy handler file
WORKDIR /app
COPY handler.py .

# Set entrypoint
CMD ["python", "handler.py"]
