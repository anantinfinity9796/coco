import pandas as pd
from utils import make_vocab, stoi, training_loop
from model import CustomResNet
from dataset import CocoDataset
from tqdm import tqdm, trange
from torchvision.transforms import Compose, Resize, Normalize, CenterCrop
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
import torch


def main():
    # Instantiate the model that will run
    # model = CustomResNet()





    # Intialize the vocabulary and populate it from items in the tokens column of the DataFrame
    vocab_dict = {}
    image_captions_df = pd.read_parquet('F://coco/captions/image_captions_df_new.parquet')
    vocab_dict = image_captions_df['tokens'].apply(lambda x : make_vocab(x,vocab_dict))

    chartoidx = {}    # create the character to index dictionary for tokens and their integer representations.
    idxtochar = []    # index to char list. Useful for decoding the output from the neural net. 
    for i,word in enumerate(vocab_dict.keys()):
        chartoidx[word] = i
        idxtochar.append(word)

    idxtochar = idxtochar + ['<unk>','<start>','<end>']    # add additional tokens
    chartoidx.update({'<unk>':22329,                       # add addtional tokens and their corresponding numbers 
                            '<start>':22330,
                            '<end>':22331})

    # Add an idx_tokens column to the dataframe to house the integer representation of the token strings.
    image_captions_df['idx_tokens'] = image_captions_df['tokens'].apply(lambda x : stoi(x, chartoidx))
    


    manual_transforms = Compose([
                            Resize(size = (256,256)),
                            CenterCrop(size=(224,224)),
                            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    

        # Create the datasets
    # train_coco = CocoDataset(image_label_df=image_label_df, val_stride=10, transforms=manual_transforms)
    # val_coco = CocoDataset(image_label_df=image_captions_df,is_val_set_bool=True, val_stride=200, transforms=manual_transforms)

    # Create the dataloaders
    # train_dataloader = DataLoader(dataset=train_coco, batch_size=128, pin_memory=True, drop_last=True, num_workers=4)
    # val_dataloader = DataLoader(dataset=val_coco, batch_size=1,pin_memory=True, drop_last = True)

    # create a test dataset and a dataloader
    test_coco = CocoDataset(image_label_df=image_captions_df, test_data_set=True, test_stride=50, transforms=manual_transforms)

    test_coco_dataloader = DataLoader(dataset=test_coco, batch_size=1, pin_memory =  True,drop_last=True)


    
    res_50 = torchvision.models.resnet50(weights =ResNet50_Weights.DEFAULT)
    cust_res = CustomResNet(chartoidx, idxtochar, pretrained_model=res_50)

    cust_res = cust_res.to(device=torch.device('cuda'))

    adam_optimizer = optim.Adam(cust_res.parameters(), lr  = 3e-2)
    training_loop(10, train_dataloader=test_coco_dataloader, val_dataloader=None, model = cust_res, optimizer=adam_optimizer)



if __name__ == '__main__':
    main()



