import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from audio_encoder import Audio_encoder,read_audio
from transformers import CLIPTextModel, CLIPTokenizer
import tensorflow as tf
from test import test_one_pic

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
        caption = self.captions[index]
        with torch.no_grad():
            text_input = tokenizer(caption, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
            text_embeddings = text_embeddings.half()
        
        try:
            audio = read_audio(audio_path)
        except:
            audio = torch.zeros_like(text_embeddings)
            text_embeddings = audio
        return audio,text_embeddings

    def __len__(self):
        return len(self.data)

#train_dataset = MyDataset('Test.csv')  
train_dataset = MyDataset('./audiocaps/new_train.csv')  
BATCH_SIZE = 16
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE)

# 定义超参数
lr = 1e-4

model = Audio_encoder(imagebind_ckpt_path='.checkpoints/imagebind_huge.pth',
                            num_audio_query_token=77)
optimizer = optim.Adam(model.parameters(), lr=lr)

#loss = torch.nn.functional.kl_div
loss = torch.nn.MSELoss()

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
        model.train()
        for i, (audio, caption) in enumerate(train_loader):
            audio = audio.to(device)
            caption = caption.to(device)
            
            outputs = model(audio)
            #loss = criterion(outputs.softmax(-1).log(), caption.softmax(-1), reduction='sum')
            loss = criterion(outputs,caption)    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, epochs, i+1, len(train_dataset)//BATCH_SIZE, loss.item()))
                with summary_writer.as_default():
                    tf.summary.scalar('train_loss', loss.item(), step=i+1)
        if valid_loader:
            model.eval()
            losses=[]
            for i, (audio, caption) in enumerate(train_loader):
                audio = audio.to(device)
                caption = caption.to(device)
                outputs = model(audio)
                loss = criterion(outputs,caption)
                losses.append(loss.item())

            val_loss = sum(losses)/len(losses)
            with summary_writer.as_default():
                    tf.summary.scalar('valid_loss', val_loss, epoch=epoch)

        test_one_pic(model,epoch)

        if (epoch+1) % 10 == 0:
            print(f'End epoch {epoch}, current loss is {loss.item()}')
    return model

if __name__ == '__main__':
    #model.load_state_dict(torch.load('audio_weights.pth'))
    summary_writer = tf.summary.create_file_writer(logdir='logs/')
    model = train(model,train_loader,loss,optimizer)
    summary_writer.close()
    torch.save(model.state_dict(), 'audio_weights_ac.pth')
    #model.load_state_dict(torch.load('audio_weights.pth'))