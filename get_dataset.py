from datasets import load_dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
from ImageBind.data import load_and_transform_audio_data

dataset = load_dataset('csv',data_files='Test.csv')
print(dataset)


def read_img(img_path):
    print(img_path)
    train_transforms = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        ]
    )
    img = Image.open(img_path)  
    img =  train_transforms(img)
    return img
    
def read_audio(audio_path):
    audio = load_and_transform_audio_data([audio_path], "cpu", clips_per_video=1)
    return audio

def preprocess(examples):
    examples["pixel_values"] = read_img(examples['image'])
    examples["input_ids"] = read_audio(examples['audio'])
    return examples
    
train_dataset = dataset["train"].map(preprocess,remove_columns=dataset["train"].column_names)
train_dataset.set_format(type='torch')
print(train_dataset['pixel_values'][0].shape)

