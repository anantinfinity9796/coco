import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
torch.random.manual_seed(42)
torch.cuda.manual_seed(42)
import pandas as pd
from tqdm import tqdm, trange
from torchvision.transforms import Compose, Resize, Normalize, CenterCrop
# from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import copy
from new_dataset import CocoDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_data():

    image_captions_df = pd.read_parquet('F://coco/captions/image_captions_df_less_than_15.parquet')


    with open('F:/coco/annotations/instances_train2014.json') as file:
        data = file.read()
        data = json.loads(data)

    category_df = pd.DataFrame.from_records(data['categories'])

    enum_super_categories = list(category_df['supercategory'].unique())
    data_df = pd.DataFrame.from_records(data['annotations'])


    data_df['category_name'] = data_df['category_id'].map(category_df.set_index('id')['name'])
    data_df['super_category'] = data_df['category_id'].map(category_df.set_index('id')['supercategory'])
    data_df['super_category_id'] = data_df['super_category'].apply(lambda x: enum_super_categories.index(x))

    new_data_df = data_df.groupby(['image_id']).agg(list)

    new_captions_df = pd.merge(left = image_captions_df, right = new_data_df, how='inner', on='image_id')

    manual_transforms = Compose([
                                Resize(size = (448,448)),
                                # CenterCrop(size=(448,448)),
    #                             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    test_coco = CocoDataset(data_df=new_captions_df,is_test = True, transforms=manual_transforms, test_percent=0.8)
    # val_coco = CocoDataset(data_df=new_captions_df,is_val = True, transforms=manual_transforms)
    # train_coco = CocoDataset(data_df=new_captions_df,is_val = True, transforms=manual_transforms)

    test_coco_dataloader = DataLoader(dataset=test_coco,shuffle=True, pin_memory=True,batch_size = 1, num_workers=2)
    # val_coco_dataloader = DataLoader(dataset=val_coco,shuffle = True, pin_memory=True,num_workers=2,batch_size = 1)
    # train_coco_dataloader = DataLoader(dataset = train_coco, shuffle = True, pin_memory=True,num_workers=2,batch_size = 1)
    return test_coco_dataloader

class YoloBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YoloBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1, bias = False)
        self.batchnorm1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size = 3, padding = 1, bias = False)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        out = self.conv1(x)
        out = F.relu(self.batchnorm1(out))

        out = self.conv2(x)
        out = F.relu(self.batchnorm2(out))
        return out
    
class Yolo(nn.Module):
    def __init__(self):

        super(Yolo,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3,padding=1,bias=False)
        self.batchnorm2 = nn.BatchNorm2d(192)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1,bias=False)
        self.batchnorm3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1,bias=False)
        self.batchnorm4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1,bias=False)
        self.batchnorm5 = nn.BatchNorm2d(256)
        
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,padding=1,bias=False)
        self.batchnorm6 = nn.BatchNorm2d(512)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv7 = nn.Sequential(YoloBlock(256,512),
                                   YoloBlock(256,512),
                                   YoloBlock(256,512),
                                   YoloBlock(256,512))
        
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1,bias=False)
        self.batchnorm8 = nn.BatchNorm2d(512)
        
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1,bias=False)
        self.batchnorm9 = nn.BatchNorm2d(1024)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv10 = nn.Sequential(YoloBlock(512,1024),
                                    YoloBlock(512,1024),)
        
        self.conv11 = nn.Conv2d(in_channels = 1024, out_channels=1024, kernel_size=3, padding=1,bias=False)
        self.batchnorm11 = nn.BatchNorm2d(1024)
        
        self.conv12 = nn.Conv2d(in_channels=1024, out_channels=1024,kernel_size=3, stride=2, padding=1,bias=False)
        self.batchnorm12 = nn.BatchNorm2d(1024)
        
        self.conv13 = nn.Conv2d(in_channels=1024, out_channels=1024,kernel_size=3, padding=1,bias=False)
        self.batchnorm13 = nn.BatchNorm2d(1024)
        
        self.conv14 = nn.Conv2d(in_channels=1024, out_channels=1024,kernel_size=3, padding=1,bias=False)
        self.batchnorm14 = nn.BatchNorm2d(1024)

        self.linear1 = nn.Linear(in_features=1024, out_features=4096,bias=False)
        self.batchnorm15 = nn.BatchNorm1d(4096)
        
        self.linear2 = nn.Linear(in_features=4096, out_features=34)
    def forward(self, x):
        out = F.relu(self.batchnorm1(self.conv1(x)))
        out = self.maxpool1(out)
        
        out = F.relu(self.batchnorm2(self.conv2(out)))
        out = self.maxpool2(out)
        
        out = F.relu(self.batchnorm3(self.conv3(out)))
        out = F.relu(self.batchnorm4(self.conv4(out)))
        out = F.relu(self.batchnorm5(self.conv5(out)))
        
        out = F.relu(self.batchnorm6(self.conv6(out)))
        out = self.maxpool3(out)
#         print(out.shape)
        out = self.conv7(out)
#         print(out.shape)
        out = F.relu(self.batchnorm8(self.conv8(out)))
        out = F.relu(self.batchnorm9(self.conv9(out)))
        out = self.maxpool4(out)
#         print(out.shape)
        out = self.conv10(out)
    
        out = F.relu(self.batchnorm11(self.conv11(out)))
        out = F.relu(self.batchnorm12(self.conv12(out)))
        out = F.relu(self.batchnorm13(self.conv13(out)))
        out = F.relu(self.batchnorm14(self.conv14(out)))
        # print(out.shape)
        out = out.reshape(-1,1024)
        # print(out.shape)
        out = F.relu(self.batchnorm15(self.linear1(out)))
        out = self.linear2(out)
        # print(out.shape)
        out = out.reshape(7,7,2,-1)
        return out
    

# Now lets code up the loss function
def calculate_loss(pred, targets):
    """This function would calculate the loss for a single image and targets tensor"""
    grid_1, grid_2, bboxes, output = pred.shape
    
    total_loss = 0
    for i in range(grid_1):
        for j in range(grid_2):
            grid_loss = 0
            # For each grid we need to find out the ground truth object with the highest IOU with out predicted bounding boxes
            # Get the two bounding boxes of the predictions
            # pred_bbox_1 = pred[i][j][0:1][:,1:5]
            pred_bbox_1 = pred[i][j][0:1]
            
            # pred_bbox_2 = pred[i][j][1:][:,1:5]
            pred_bbox_2 = pred[i][j][1:]
            
            max_iou = -99 
            
            max_iou_pred_box = None
            max_iou_target_box = None
            # Now we will calulate the iou between each predicted bounding box and each of the bounding boxes of the image.
            for box in targets:
                # Convert the bounding box to the correct format
                box = box.unsqueeze(0)
                converted_bbox = torchvision.ops.box_convert(box[:,1:5], in_fmt='xywh', out_fmt='xyxy')
                
                # Calculate the iou of both of the predicted bounding boxes with  each box to get the highest iou
                pred_bbox_1_iou = torchvision.ops.box_iou(pred_bbox_1[:,1:5], converted_bbox)
                pred_bbox_2_iou = torchvision.ops.box_iou(pred_bbox_2[:,1:5], converted_bbox)

                # Now get the target bbox with the highest_iou with our bboxes.
                if pred_bbox_1_iou > max_iou or pred_bbox_2_iou > max_iou:
                    if pred_bbox_1_iou > pred_bbox_2_iou:
                        max_iou = pred_bbox_1_iou
                        max_iou_pred_box = pred_bbox_1
                        max_iou_target_box = box
                    else:
                        max_iou = pred_bbox_2_iou
                        max_iou_pred_box =  pred_bbox_2
                        max_iou_target_box = box
            
            # Now we have the predicted box with the max iou in max_iou_pred_box and target box with the max iou in max_iou_target_box.
            # Next we will need to calculate the losses of each component of confidence score, box_loss, class_loss
            # First up is the confidence loss
            
            if max_iou > 0.50:
                # This means that the network is confident that there is an object
                confidence_loss = F.mse_loss(max_iou_pred_box[:,0:1]*max_iou, max_iou_target_box[:,0:1])
            else:
                confidence_loss = 0.5 * F.mse_loss(max_iou_pred_box[:,0:1]*max_iou, max_iou_target_box[:,0:1])
            
            # Now the box loss
            bbox_loss = 5* F.mse_loss(max_iou_pred_box[:,1:5], max_iou_target_box[:,1:5])

            # Then we need to calculate the class_loss
            class_loss = F.mse_loss(max_iou_pred_box[:,5:], max_iou_target_box[:,5:])

            # calculate the loss of each individual grid and normalize it with the number of grids
            grid_loss = (grid_loss + confidence_loss + bbox_loss + class_loss)/7.0


        total_loss += grid_loss
    return total_loss



            
def training_loop(epochs, model, train_dataloader, optimizer, ):
    for epoch in trange(epochs):
        epoch_loss = 0
        for batch in tqdm(train_dataloader):
            # print(batch)
            image, targets, cat_name, image_id = batch

            image = image.to(device)
            targets = targets.squeeze(0).to(device)
            # print(targets.shape)

            pred = model(image)
            # print(pred)

            optimizer.zero_grad()

            loss = calculate_loss(pred, targets)

            

            loss.backward()

            optimizer.step()
            epoch_loss += loss
        print(f"Epoch : {epoch} | Loss : {epoch_loss/len(train_dataloader)} ")





if __name__ == '__main__':
    yolo_model = Yolo()
    yolo_model = yolo_model.to(device = torch.device('cuda'))

    adam_optimizer = optim.Adam(yolo_model.parameters(), lr = 3e-3)

    test_coco_dataloader = process_data()
    training_loop(10, yolo_model, test_coco_dataloader, adam_optimizer)