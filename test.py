from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

config = OmegaConf.load('audio_config.yaml')
model = instantiate_from_config(config['model'])

print(model)