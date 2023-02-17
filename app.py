import os
import json
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"


# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
  pipe = StableDiffusionPipeline.from_pretrained(model_id)
  pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
  pipe = pipe.to("cpu")
  # pipe.enable_attention_slicing()

  image = pipe(prompt, height=512, width=512, num_inference_steps=80).images[0]
      
  image.save("astronaut_rides_horse.png")