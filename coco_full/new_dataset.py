import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision
torch.random.manual_seed(42)
torch.cuda.manual_seed(42)
import pandas as pd
import torchvision
import torch.nn.functional as F
import copy

class CocoDataset(Dataset):
    def __init__(self,data_df, is_train=False, is_val=False, is_test = False, val_percent=10, test_percent = 5, transforms = None):
        self.new_data_df = data_df.copy()
        self.transforms = transforms
        
        
        
        if is_val:
            self.data_df = self.new_data_df.sample(frac = float(val_percent/100), axis = 0, ignore_index = True)
        if is_test:
            self.data_df = self.new_data_df.sample(frac = float(test_percent/100), axis = 0, ignore_index = True)
        else:
            self.data_df = self.new_data_df.sample(frac = 1- (float(val_percent/100) + float(test_percent/100)), axis = 0, ignore_index = True)
            
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, ndx):
        folder_path = 'F://coco/train2014/train2014/'
        
        row = self.data_df.iloc[ndx]
        file_name = row['file_name']
        bboxes = torch.tensor(row['bbox'])
        cat_id = torch.tensor(row['super_category_id'])
        cat_names = row['super_category']
        image_id = row['image_id']

#             captions_array = torch.tensor(row['shortest_idx_tokens'][0][0])

        # Get the image-data from file_name
        image_array = torchvision.io.read_image(folder_path + file_name)
        image_array = (image_array/255.0).to(torch.float32)
        # if self.transforms != None:
            # image_array = self.transforms(image_array).to(torch.float32)
        
        # After transforming the image we convert the individual bounding boxes, classes and confidence of object to target tensor
        # The format of the out tensor is [object_confidence, bounding_boxes, one-hot class labels]
        y = torch.full((len(bboxes),1), fill_value = 1)
        new = torch.cat((y,bboxes), axis = 1)
        x = F.one_hot(cat_id, num_classes = 12)
        out = torch.cat((new,x ), axis = 1)
        
        return image_array, out, cat_names, image_id
        
    def get_data_image_id(self,image_id=None):
        """ This function returns an image by its id and the corresponding multiple labesl in a present/no_present binary format"""
        

        folder_path = 'F://coco/train2014/train2014/'

        if image_id == None:
            raise ValueError('Must provide IMAGE ID')
        
        else:
            row = self.data_df[self.data_df['image_id']==image_id]
            file_name = row['file_name'].values[0]
            bounding_boxes_list = row['bbox'].values[0]
            category_id_list = row['category_id'].values[0]
            category_names_list = row['category_name'].values[0]

#             captions_array = torch.tensor(row['shortest_idx_tokens'][0][0])

            # Get the image-data from file_name
            image_array = torchvision.io.read_image(folder_path + file_name)
            image_array = (image_array/255.0).to(torch.float32)
            # if self.transforms != None:
                # image_array = self.transforms(image_array).to(torch.float32)
                
            # After transforming the image now we set up the target tensor that will be outputted
            # After transforming the image now we set up the target tensor that will be outputted
            y = torch.full((len(category_id_list),1), fill_value = 1)
            new = torch.cat((y,bounding_boxes_list), axis = 1)
            x = F.one_hot(category_id_list, num_classes = 12)
            out = torch.cat((new,x), axis = 1)
            return image_array, out, category_names_list
        
            