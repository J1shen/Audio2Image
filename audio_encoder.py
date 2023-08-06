import contextlib
import torch
import torch.nn as nn
from transformers import BertConfig
from Qformer import BertConfig, BertLMHeadModel
from ImageBind.models.imagebind_model import ImageBindModel,ModalityType
from ImageBind.models import imagebind_model
from ImageBind.data import load_and_transform_audio_data

text_list=["A dog.", "A car", "A bird"]
image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
audio_paths=["ImageBind/.assets/dog_audio.wav", "ImageBind/.assets/car_audio.wav", "ImageBind/.assets/bird_audio.wav"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Audio_encoder(nn.Module):
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    def init_video_Qformer(self, num_query_token, vision_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.is_decoder = True
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    def __init__(self,
                 imagebind_ckpt_path,
                 num_audio_query_token = 8,
                 proj_size = 768,
                 frozen_audio_Qformer = False) -> None:
            super().__init__()
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print (f'Initializing audio encoder from {imagebind_ckpt_path} ...')
            self.audio_encoder,self.audio_hidden_size = imagebind_model.imagebind_huge()
            self.audio_encoder.load_state_dict(torch.load(imagebind_ckpt_path))
            # free vision encoder
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
            print ('Audio encoder initialized.')
            
            self.num_audio_query_token = num_audio_query_token
            self.audio_Qformer,self.audio_query_tokens = self.init_video_Qformer(
                num_query_token = self.num_audio_query_token,
                vision_width=self.audio_hidden_size,
                num_hidden_layers=2)
            self.audio_Qformer.cls = None
            self.audio_Qformer.bert.embeddings.word_embeddings = None
            self.audio_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.audio_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.audio_proj = nn.Linear(
                self.audio_Qformer.config.hidden_size, proj_size
            )
            self.audio_position_embedding = nn.Embedding(num_audio_query_token, self.audio_hidden_size)

            if frozen_audio_Qformer:
                #  todo frozen  llama_proj
                for name, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = False
                self.audio_query_tokens.requires_grad = False
                for name, param in self.audio_proj.named_parameters():
                    param.requires_grad = False
                for name, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = False
                print('Audio_Qformer and audio-LLAMA proj is frozen')
            else:
                for name, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = True
                self.audio_query_tokens.requires_grad = True
                for name, param in self.audio_proj.named_parameters():
                    param.requires_grad = True
                for name, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = True
                print('Audio_Qformer is not frozen')
                
    def forward(self, audio, modality_type=ModalityType.AUDIO):
        device = audio.device
        with self.maybe_autocast():
            audio_feature, audio_imagebind_finalout = self.audio_encoder.get_audio_feature(audio,modality_type=modality_type)
            batch_size,time_length = audio.size()[:2]


            position_ids = torch.arange(time_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            audio_position_embeddings = self.audio_position_embedding(position_ids)
            audio_imagebind_finalout = audio_imagebind_finalout + audio_position_embeddings

            audio_query_tokens = self.audio_query_tokens.expand(audio_imagebind_finalout.shape[0], -1, -1)
            frame_atts = torch.ones(audio_imagebind_finalout.size()[:-1], dtype=torch.long).to(device)

            audio_query_output = self.audio_Qformer.bert(
                query_embeds=audio_query_tokens, #[32,768]
                encoder_hidden_states=audio_imagebind_finalout,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            audio_hidden = audio_query_output.last_hidden_state

            inputs_llama = self.audio_proj(audio_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
    
        return inputs_llama

if __name__ == '__main__':
    encoder = Audio_encoder(imagebind_ckpt_path='.checkpoints/imagebind_huge.pth',
                            num_audio_query_token=1)
    encoder.to(device)
    audio = load_and_transform_audio_data([audio_paths[0]],"cpu", clips_per_video=1)
    audio = audio.to(device)
    #print(audio)
    audio = audio.squeeze(0)
    print(audio.shape)
    
    result = encoder(audio)
    #print(result)
    print(result.shape)