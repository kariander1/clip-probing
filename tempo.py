import torch
import random
from diffusers import StableDiffusionPipeline

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

seed = 45
set_seed(seed)

# Load the Stable Diffusion 1.5 model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Use CUDA if available

# Prompt for image generation
prompt = "a photograph of a spoon on a toilet"

# Generate the image
with torch.autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5).images[0]

# Save the image
image.save("generated_image.png")

print("Image generated and saved as 'generated_image.png'")
