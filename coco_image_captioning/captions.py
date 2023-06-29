import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.optim as optim
import torch.nn as nn
from torchinfo import summary
from tqdm import tqdm, trange
torch.random.manual_seed(42)
torch.cuda.manual_seed(42)
from torchvision.transforms import Compose, Resize, Normalize, CenterCrop
from torchvision.models import resnet50, ResNet50_Weights

def make_vocab(token_list):
    for tokens in  token_list:
        for token in tokens:
            if token in vocab_dict:
                vocab_dict[token] +=1
            else:
                vocab_dict[token] = 1



def stoi(captions_list):
    caption_token_list = []
    for caption in captions_list:
        tokens_list = [chartoidx['<start>']] + []
        for token in caption:
            try:
                tokens_list.append(chartoidx[token])
            except:
                tokens_list.append(chartoidx['<unk>'])
        tokens_list = tokens_list + [chartoidx['<end>']]
        caption_token_list.append(tokens_list)
    return caption_token_list


vocab_dict = {}
image_captions_df = pd.read_parquet('F://coco/captions/image_captions_df_new.parquet')
image_captions_df['tokens'].apply(lambda x : make_vocab(x))

chartoidx = {}
idxtochar = []
for i,word in enumerate(vocab_dict.keys()):
    chartoidx[word] = i
    idxtochar.append(word)

idxtochar = idxtochar + ['<unk>','<start>','<end>']
chartoidx.update({'<unk>':22329,
                         '<start>':22330,
                         '<end>':22331})

image_captions_df['idx_tokens'] = image_captions_df['tokens'].apply(lambda x : stoi(x))

class CocoDataset(Dataset):
    def __init__(self,image_label_df, val_stride = 10, is_val_set_bool = False, test_data_set =False, test_stride = 5, transforms = None):
        
        self.transfrom = transforms
        self.is_val_set_bool = is_val_set_bool
        self.val_stride = val_stride
        self.image_label_df = image_label_df.copy()
        self.test_data_set = test_data_set
        self.test_stride = test_stride

        if self.test_data_set: #If we need only a small subset of the data to work with
            self.image_label_df = self.image_label_df[::test_stride].reset_index(drop = True)

        elif self.is_val_set_bool: # If we need only the validation data then return the validation data which is a subset of total data
            assert self.val_stride > 0
            self.image_label_df = self.image_label_df[::val_stride]

        elif self.val_stride > 0:  # Else if val_stride is greater than zero then return the remaining dataframe after removing 10% of data
            self.image_label_df = self.image_label_df.drop(index = list(range(0,len(self.image_label_df),self.val_stride)))
            
        else: # else train on the full dataset
            self.image_label_df = self.image_label_df
        
    def __len__(self):
        """ This method calculates the length of your data."""
        return len(self.image_label_df)

    # Now we will introduce function which gives us an image and its corresponding labels by id
    def get_image_by_id(self, img_id = None):
        """ This function returns an image by its id and the corresponding multiple labesl in a present/no_present binary format"""
        

        folder_path = 'F://coco/train2014/train2014/'

        if img_id == None:
            raise ValueError('Must provide IMAGE ID')

        else:
            row = self.image_label_df[self.image_label_df['image_id']==img_id]
            file_name = row['file_name'].values[0]

            captions_array = torch.tensor(row['idx_tokens'][0][0])

            # Get the image-data from file_name
            image_array = torchvision.io.read_image(folder_path + file_name)
            image_array = (image_array/255.0).to(torch.float32)
            image_array = self.transfrom(image_array).to(torch.float32)
        
            return image_array, captions_array


    # Now we would write the code for returning the image from index and its corresponding labels. Which would be used by the train/val dataloaders
    def __getitem__(self, ndx):

        """ This function takes in an index and returns the image and the labels of the image at that index"""
        folder_path = 'F://coco/train2014/train2014/'
        
        row = self.image_label_df.iloc[ndx]
        # print(row)

        # Now get the image_id and the file_name of the image
        image_id = row['image_id']
        file_name = row['file_name']
        
        # Now get the captions array from the dataframe
        captions_array = torch.tensor(row['idx_tokens'][0])
        # Now get the image_data from storage
        image_array = torchvision.io.read_image(folder_path + file_name)
        image_array = (image_array/255.0).to(torch.float32)
        image_array = self.transfrom(image_array).to(torch.float32)
        # print(captions_array)
        return (image_array, captions_array, image_id)


class CustomResNet(nn.Module):
    def __init__(self, pretrained_model = None):
        super().__init__()



        for parameter in pretrained_model.parameters():
            parameter.requires_grad = False

        
        pretrained_model.fc = nn.Linear(2048,256)

        
        self.backbone = nn.Sequential(pretrained_model)
        # Till the previous step we have a pretrained Resnet whose weights we have frozen. Now we will define the RNN

        # self.embedding_matrix = torch.randn(len(chartoidx), 50, requires_grad=True) / (len(chartoidx)**0.5)
        self.embedding_matrix = nn.Embedding(len(chartoidx), 50)
        self.rnn = nn.RNNCell(input_size=50, hidden_size = 256, nonlinearity='tanh')
        self.output_layer = nn.Linear(256,len(chartoidx))

        self.cross_entopy_loss = nn.CrossEntropyLoss()

    def forward(self, image, caption_list):
        out_backbone = self.backbone(image)

        hidden = out_backbone.squeeze(0)
        # print(f"the hidden shape from image is {hidden.shape}")
        i = 0
        # new_char_id = char_id
        loss_word_list = []
        current_word_id = caption_list[i]
        target_index = caption_list[i+1]
        while True:
            
            if i == 0:
                current_word_id = caption_list[0]
            target_index = caption_list[i+1]
            if idxtochar[target_index] == '<end>':
                break
            out_embedding = self.embedding_matrix(current_word_id)  # embedding_shape = (1,50)
            # print(out_embedding.shape)
            # print(out_embedding, out_embedding.requires_grad)
            # print(f"The out embedding shape is {out_embedding.shape}")
            # calculate the hidden state activation from the embedding and the previous hidden state and get the next hidden state
            hidden = self.rnn(out_embedding, hidden)   # hidden_shape = (1,1024)

            # Now dish out the output_logits of the current time step
            output_logits = self.output_layer(hidden)   # output_logits shape (1,len(vocab))
            # print(f"The shape of output logits is {output_logits.shape}")

            # # calculate the probabilities of the characters
            probs =  torch.softmax(output_logits, dim = 0)         # probs_shape = (1,len(vocab))

            # print(f"The shape of probs is {probs.shape}")
            # print(f"The shape of output probabilities is {probs.shape}")
            # # Now get the index of the larget probability.
            next_word_id = torch.argmax(probs, dim =0)
            # print(f"the next word id  is {idxtochar[next_word_id.item()]}, and id is {next_word_id.item()}")

            # Now we will take the next_word that the cell thinks will be and do two things.
            # First we use it and the target to calculate the loss
            current_word_loss = self.cross_entopy_loss(probs, caption_list[i+1])
            # append the current_word_loss to loss_word_list
            loss_word_list.append(current_word_loss)
            
            # Second we would use the index of the current most probable output word as an input of the next time-step
            current_word_id = next_word_id
            # increase the value of i by 1 so that we can get the target word for the next input word
            i+=1
            
        final_loss = torch.sum(torch.tensor(loss_word_list, requires_grad=True))
        return final_loss


# Now lets create the training loop which would take in the image and the caption train on it word by word
def training_loop(epochs, train_dataloader, val_dataloader, model, optimizer):
    for epoch in range(epochs):
        loss_list = []
        for batch in tqdm(train_dataloader):
            image, caption, _ = batch
            # print(image.shape)
            # image = torch.unsqueeze(image, dim=0)
            # print(caption.shape)
            # image = image.to(device = torch.device('cuda')),.,
            # caption = caption.to(device = torch.device('cuda'))
            optimizer.zero_grad()
            output =  model(image, caption[0])
            # print(output, grad_info)
            # break
            loss_list.append(output.item())
            output.backward()
            # print(grad_b)
            optimizer.step()
            # print(output.item())
        print(f"The loss is {sum(loss_list)/len(loss_list)}")










def main():
    manual_transforms = Compose([
                            Resize(size = (256,256)),
                            CenterCrop(size=(224,224)),
                            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    

        # Create the datasets
    # train_coco = CocoDataset(image_label_df=image_label_df, val_stride=10, transforms=manual_transforms)
    val_coco = CocoDataset(image_label_df=image_captions_df,is_val_set_bool=True, val_stride=200, transforms=manual_transforms)

    # Create the dataloaders
    # train_dataloader = DataLoader(dataset=train_coco, batch_size=128, pin_memory=True, drop_last=True, num_workers=4)
    val_dataloader = DataLoader(dataset=val_coco, batch_size=1,pin_memory=True, drop_last = True)

    # create a test dataset and a dataloader
    test_coco = CocoDataset(image_label_df=image_captions_df, test_data_set=True, test_stride=50, transforms=manual_transforms)

    test_coco_dataloader = DataLoader(dataset=test_coco, batch_size=1, pin_memory =  True,drop_last=True)



    res_50 = torchvision.models.resnet50(weights =ResNet50_Weights.DEFAULT)
    cust_res = CustomResNet(pretrained_model=res_50)


    adam_optimizer = optim.Adam(cust_res.parameters(), lr  = 3e-2)
    training_loop(10, train_dataloader=test_coco_dataloader, val_dataloader=None, model = cust_res, optimizer=adam_optimizer)


if __name__ == "__main__":
    main()