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
from torch.utils.tensorboard import SummaryWriter
import datetime as dt
# from resnet_script.coco_dataset import CocoDataset, training_loop, validation_loop, CustomResNet

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

            
            label_array = torch.zeros(len(self.image_label_df.columns[2:]),2)  # create an array of zeros which is a (num_classes, 2) array

            # Lets get the multilabel array for a particular image_id from image_label_df
            multilabel_array = row.values[0][2:]

            # Now lets populate the label_array with values from the multilabel array
            label_array[range(len(multilabel_array)), multilabel_array] = 1

            # Get the image-data from file_name
            image_array = torchvision.io.read_image(folder_path + file_name)
            image_array = (image_array/255.0).to(torch.float32)
            image_array = self.transfrom(image_array).to(torch.float32)
        
            return image_array, label_array


    # Now we would write the code for returning the image from index and its corresponding labels. Which would be used by the train/val dataloaders
    def __getitem__(self, ndx):

        """ This function takes in an index and returns the image and the labels of the image at that index"""
        folder_path = 'F://coco/train2014/train2014/'
        
        row = self.image_label_df.iloc[ndx]

        # Now get the image_id and the file_name of the image
        image_id = row['image_id']
        file_name = row['file_name']

        # print(image_id, file_name)
        # Now get the multilabel in the form of a numpy array
        # This array would be of the shape (num_classes, 2). Which means that if a class is present or not present and we will keep 1 at that
        # Input probabilities for each class wo
        label_array = torch.zeros(len(self.image_label_df.columns[2:]),2)  # create an array of zeros which is a (num_classes, 2) array


        # Now we will use the multiple_labels array to populate the categories which are present in the picture.
        # For each category we will populate 1 at the 0th position if it is not present or 1 at the 1th position if it is present
        # This way we can check for the prescence or absence of multiple categories which makes it a multilabel classification problem

        

        # Lets get the multilabel array for a particular image_id
        multilabel_array = row.values[2:].astype(np.float32)

        # Now lets populate the label_array with values from the multilabel array
        label_array[range(len(multilabel_array)), multilabel_array] = 1

        # Now get the image_data from storage
        image_array = torchvision.io.read_image(folder_path + file_name)
        image_array = (image_array/255.0).to(torch.float32)
        image_array = self.transfrom(image_array).to(torch.float32)
        return (image_array, label_array, image_id)


# Now we will begin to write the training loop that will ingest the data and output the result.
def training_loop(epochs, conv_model,train_dataloader,device,optimizer= None, val_dataloader=None, loss_fn = None, schedular = None, writer = None):
    

    for epoch in trange(epochs):
        print("Training is progressing", end = '\n')
        train_losses = []
        # print("Training is Progressing")
        for img, label,_ in tqdm(train_dataloader):
            batch_losses = []
            img = img.to(device)
            label = label.to(device)

            # Now put the training data into the model to get the logits
            pred = conv_model(img)

            # Now we will zero out the optimizer
            optimizer.zero_grad()
            
            # calculate the loss
            loss = loss_fn(pred, label)

            out_loss = loss.clone().to(device=torch.device('cpu'))

            # backpropagate
            loss.backward()
            # Next we will update the parameters
            optimizer.step()
            # print(out_loss.item())
            train_losses.append(out_loss.item())
            # print(f"Train for the batch is {out_loss.item()}")
        print("Validation is progressing", end = '\n')
        val_loss_list = validation_loop(model = conv_model, val_dataloader=val_dataloader, device = device, loss_fn= loss_fn)
        print(f"Epoch: {epoch} | Train_loss: {torch.tensor(train_losses).mean().item()}) | Val_loss : {torch.tensor(val_loss_list).mean().item()}")

        # step the learning rate schedular
        schedular.step()



        # Writing the training and the validation losses to tensorboard.
        writer.add_scalar(tag = 'Training Loss Per Epoch',scalar_value=torch.tensor(train_losses).mean().item(), global_step = epoch)
        writer.add_scalar(tag = 'Validation Loss Per Epoch',scalar_value=torch.tensor(val_loss_list).mean().item(), global_step = epoch)
        # break
            

    return


def validation_loop(model, val_dataloader, device, loss_fn):
    # print("Validation is progressing")
    val_loss_list = []
    model.eval()
    with torch.no_grad():
        for val_img, val_label, _ in tqdm(val_dataloader, nrows=1):
            # transfer the images and labels to the GPU
            val_img = val_img.to(device)
            val_label = val_label.to(device)

            # Now run it through the model and get the val_logits
            val_logits = model(val_img)

            val_loss = loss_fn(val_logits, val_label)
            out_val_loss = val_loss.clone().to(torch.device('cpu'))
            val_loss_list.append(out_val_loss.item())
            # val_loss += out_val_loss.item()
            # val_i +=1
    return val_loss_list
            

class CustomResNet(nn.Module):
    def __init__(self, pretrained_model = None):
        super().__init__()



        for parameter in pretrained_model.parameters():
            parameter.requires_grad = False


        pretrained_model.fc = nn.Linear(2048,1024)
        # pretrained_model.b0 = nn.BatchNorm1d(1024, track_running_stats=True)
        self.backbone = nn.Sequential(pretrained_model)

        self.fc1 = nn.Linear(1024,512)
        self.b1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(512,128)
        self.b2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        # Now we will need to define our output layers
        # computation for the multi-label heads
        
        self.head1 = nn.Linear(128,2)
        self.sig1 = nn.Sigmoid()
        self.head2 = nn.Linear(128,2)
        self.sig2 = nn.Sigmoid()
        self.head3 = nn.Linear(128,2)
        self.sig3 = nn.Sigmoid()
        self.head4 = nn.Linear(128,2)
        self.sig4 = nn.Sigmoid()
        self.head5 = nn.Linear(128,2)
        self.sig5 = nn.Sigmoid()
        self.head6 = nn.Linear(128,2)
        self.sig6 = nn.Sigmoid()
        self.head7 = nn.Linear(128,2)
        self.sig7 = nn.Sigmoid()
        self.head8 = nn.Linear(128,2)
        self.sig8 = nn.Sigmoid()
        self.head9 = nn.Linear(128,2)
        self.sig9 = nn.Sigmoid()
        self.head10 = nn.Linear(128,2)
        self.sig10 = nn.Sigmoid()
        self.head11 = nn.Linear(128,2)
        self.sig11 = nn.Sigmoid()
        self.head12 = nn.Linear(128,2)
        self.sig12 = nn.Sigmoid()



    def forward(self, X):
        out_backbone = self.backbone(X)

        out  = self.relu1(self.b1(self.fc1(out_backbone)))
        out = self.relu2(self.b2(self.fc2(out)))

        
        # Now we will define the computations of the 12 heads and heads and put all the outputs into one tensor and return that tensor.
        self.out_head1 = self.sig1(self.head1(out))  # 'food'
        self.out_head2 = self.sig2(self.head2(out)) # animal
        self.out_head3 = self.sig3(self.head3(out)) # furniture
        self.out_head4 = self.sig4(self.head4(out)) # electronic
        self.out_head5 = self.sig5(self.head5(out)) # kitchen
        self.out_head6 = self.sig6(self.head6(out)) # vehicle
        self.out_head7 = self.sig7(self.head7(out)) # person
        self.out_head8 = self.sig8(self.head8(out)) # outdoor
        self.out_head9 = self.sig9(self.head9(out)) # accessory
        self.out_head10 = self.sig10(self.head10(out)) # sports
        self.out_head11 = self.sig11(self.head11(out)) # appliance
        self.out_head12 = self.sig12(self.head12(out)) # indoor
        out_list = [self.out_head1,self.out_head2,self.out_head3,self.out_head4,self.out_head5,self.out_head6,self.out_head7,self.out_head8,
                    self.out_head9,self.out_head10,self.out_head11,self.out_head12]

        out_tensor = torch.stack(out_list, dim = 1)
        return out_tensor






def main():
    # Initializing global variables
    time = dt.datetime.now().strftime('%Y%m%d%H%M%S')
    folder_string = f'../runs/resnet_exp_2_pretrained_{time}'
    print(folder_string)
    writer = SummaryWriter(folder_string)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    manual_transforms = Compose([
                            Resize(size = (256,256)),
                            CenterCrop(size=(224,224)),
                            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    

    # auto_transforms = ResNet50_Weights.DEFAULT.transforms()
    image_label_df = pd.read_parquet('F://coco/captions/image_labels.parquet')


    # Create the datasets
    # train_coco = CocoDataset(image_label_df=image_label_df, val_stride=10, transforms=manual_transforms)
    val_coco = CocoDataset(image_label_df=image_label_df,is_val_set_bool=True, val_stride=20, transforms=manual_transforms)

    # Create the dataloaders
    # train_dataloader = DataLoader(dataset=train_coco, batch_size=128, pin_memory=True, drop_last=True, num_workers=4)
    val_dataloader = DataLoader(dataset=val_coco, batch_size=128,pin_memory=True, drop_last = True, num_workers=4)

    # create a test dataset and a dataloader
    test_coco = CocoDataset(image_label_df=image_label_df, test_data_set=True, test_stride=10, transforms=manual_transforms)

    test_coco_dataloader = DataLoader(dataset=test_coco, batch_size=128, pin_memory=True, drop_last=True, num_workers=4)

    # Import the custome resnet model
    res_50 = torchvision.models.resnet50(weights =ResNet50_Weights.DEFAULT)
    print(summary(res_50, input_size = (32,3,224,224)))
    # create an instance of your custome resent model
    custom_model = CustomResNet(pretrained_model=res_50)

    # Now we will move the model to the GPU and define optimizer and move all the other things to the GPU also
    custom_model = custom_model.to(device = device)
    bce_loss = torch.nn.BCELoss()
    optimizer_adam = optim.Adam(custom_model.parameters(), lr = 3e-4, weight_decay=0.001)
    scehdular = optim.lr_scheduler.StepLR(optimizer= optimizer_adam, gamma = 0.1, step_size = 10, verbose = True )


    # Next we will implement the training loop and the validation loops which will give out the results.
    # training_loop(
    #     epochs = 30,
    #     conv_model=custom_model,
    #     train_dataloader=test_coco_dataloader,
    #     device = device,
    #     optimizer=optimizer_adam,
    #     loss_fn=bce_loss,
    #     val_dataloader=val_dataloader,
    #     schedular = scehdular,
    #     writer = writer
    # )

if __name__=='__main__':
    main()

    
