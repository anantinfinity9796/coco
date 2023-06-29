import torch
from torch.utils.data import Dataset
import torchvision
torch.random.manual_seed(42)
torch.cuda.manual_seed(42)




class CocoDataset(Dataset):
    def __init__(self,image_label_df, val_stride = 10, is_val_set_bool = False, test_data_set =False, test_stride = 5, transforms = None):
        
        self.transfrom = transforms
        # self.is_val_set_bool = is_val_set_bool
        # self.val_stride = val_stride
        self.image_label_df = image_label_df.copy()
        # self.test_data_set = test_data_set
        # self.test_stride = test_stride

        # if self.test_data_set: # Making the test set
        #     self.image_label_df = self.image_label_df[::test_stride].reset_index(drop = True) # Get the data
        #     # Now remove the indexes from the dataframe the we used.
        #     self.image_label_df = self.image_label_df.drop(index = list(range(0,len(self.image_label_df),self.test_stride)))

        # elif self.is_val_set_bool: # If we need only the validation data then return the validation data which is a subset of total data
        #     assert self.val_stride > 0
        #     self.image_label_df = self.image_label_df[::val_stride]

        # elif self.val_stride > 0:  # Else if val_stride is greater than zero then return the remaining dataframe after removing 10% of data
        #     self.image_label_df = self.image_label_df.drop(index = list(range(0,len(self.image_label_df),self.val_stride)))
            
        # else: # else train on the full dataset
        #     self.image_label_df = self.image_label_df
        
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

            captions_array = torch.tensor(row['shortest_idx_tokens'][0][0])

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
        captions_array = torch.tensor(row['shortest_idx_tokens'])
        # Now get the image_data from storage
        image_array = torchvision.io.read_image(folder_path + file_name)
        image_array = (image_array/255.0).to(torch.float32)
        image_array = self.transfrom(image_array).to(torch.float32)
        # print(captions_array)
        return (image_array, captions_array, image_id)