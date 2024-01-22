from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained('./models/diffusion/stable-diffusion-v1-5', torch_dtype=torch.float32)
pipe.enable_sequential_cpu_offload()
pipe.enable_vae_slicing()
pipe.unet.to(memory_format=torch.channels_last)
pipe.enable_attention_slicing(1)
pipe.safety_checker = None

prompt = 'grasp the cup, side view'

image = pipe(prompt).images[0]

image.save('./test_side.png')