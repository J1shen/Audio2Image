import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from audio_encoder import Audio_encoder,read_audio
from transformers import CLIPTextModel, CLIPTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
text_encoder.requires_grad_ = False
text_encoder.eval()

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = pd.read_csv(data)
        self.captions = self.data['caption']
        self.audios = self.data['audio']

    def __getitem__(self, index):
        audio_path = self.audios[index]
        audio = read_audio(audio_path)
        caption = self.captions[index]
        with torch.no_grad():
            text_input = tokenizer(caption, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        return audio,text_embeddings

    def __len__(self):
        return len(self.data)

train_dataset = MyDataset('Test.csv')   
train_loader = DataLoader(train_dataset)

# 定义超参数
lr = 1e-4
num_epochs = 1000

model = Audio_encoder(imagebind_ckpt_path='.checkpoints/imagebind_huge.pth',
                            num_audio_query_token=77)
optimizer = optim.Adam(model.parameters(), lr=lr)

loss = torch.nn.functional.kl_div
torch.nn.KLDivLoss()
#loss = torch.nn.MSELoss()

def train(model,
          train_loader,
          criterion,
          optimizer,
          epochs=100,
          valid_loader=None,
          ):
    
    model.to(device)
    for epoch in range(epochs):
        print(f'Begin epoch {epoch}')
        for i, (audio, caption) in enumerate(train_loader):
            audio = audio.to(device)
            caption = caption.to(device)
            
            outputs = model(audio)
            loss = criterion(outputs.softmax(-1).log(), caption.softmax(-1), reduction='sum')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, len(train_dataset)//32, loss.item()))
                
        if (epoch+1) % 50 == 0:
            print(f'End epoch {epoch}, current loss is {loss.item()}')
    return model

if __name__ == '__main__':
    #model.load_state_dict(torch.load('audio_weights.pth'))
    model = train(model,train_loader,loss,optimizer,num_epochs)
    torch.save(model.state_dict(), 'audio_weights.pth')
    #model.load_state_dict(torch.load('audio_weights.pth'))