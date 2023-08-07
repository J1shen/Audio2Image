from ImageBind.models.imagebind_model import ImageBindModel,ModalityType
from ImageBind.models import imagebind_model
from ImageBind.data import load_and_transform_audio_data
import torch
from diffusers import StableUnCLIPImg2ImgPipeline

# construct models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16
).to(device)
imagebind_ckpt_path = '.checkpoints/imagebind_huge.pth'
model,_ = imagebind_model.imagebind_huge()
model.load_state_dict(torch.load(imagebind_ckpt_path))
model.eval().to(device)

# generate image
with torch.no_grad():
    audio_paths=["ImageBind/.assets/dog_audio.wav"]
    embeddings = model.forward({
        ModalityType.AUDIO: load_and_transform_audio_data(audio_paths, device),
    })
    embeddings = embeddings[ModalityType.AUDIO]
    print(embeddings.shape)
    images = pipe(image_embeds=embeddings.half()).images
    images[0].save("audio2img.png")