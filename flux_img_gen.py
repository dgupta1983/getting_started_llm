import torch
from diffusers import Flux2KleinPipeline

device = "mps"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=dtype,use_safetensors=True)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

prompt = "A panda holding a sign that says hello world"
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device="cpu").manual_seed(0)
).images[0]
image.save("flux-klein.png")
