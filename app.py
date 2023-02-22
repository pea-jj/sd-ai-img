import sys
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config)
pipe = pipe.to("cuda")
# pipe.enable_attention_slicing()

image = pipe("Bald man wearing backless bikini, back", height=512, width=512,
             num_inference_steps=400).images[0]

image.save("astronaut_rides_horse.png")