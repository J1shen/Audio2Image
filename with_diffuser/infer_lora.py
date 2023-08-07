import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from accelerate import Accelerator
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from audio_encoder import Audio_encoder
from ImageBind.data import load_and_transform_audio_data
from transformers import CLIPTextModel, CLIPTokenizer

vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4' , subfolder="vae", revision=None)
audio_encoder = Audio_encoder(imagebind_ckpt_path='.checkpoints/imagebind_huge.pth',
                            num_audio_query_token=1)
unet = UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="unet", revision=None)
text_encoder = CLIPTextModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="text_encoder", revision=None)
pipeline = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4',
        vae=vae,
        unet=unet,
        safety_checker=None,
        revision=None,
        torch_dtype=torch.float16,
    )
pipeline.text_encoder=audio_encoder
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config, use_karras_sigmas=True
)
#pipeline.load_lora_weights(".", weight_name="light_and_shadow.safetensors")
def read_audio(audio_path):
    audio = load_and_transform_audio_data([audio_path], "cpu", clips_per_video=1)
    audio = audio.squeeze(0)        
    return audio

audio = read_audio('ImageBind/.assets/bird_audio.wav')
images = pipeline(prompt=audio, 
    negative_prompt='', 
    width=512, 
    height=768, 
    num_inference_steps=15, 
    num_images_per_prompt=4,
    generator=torch.manual_seed(0)
).images

print(images)