from tqdm import tqdm
import torch
def make_vocab(token_list,vocab_dict):
    """ This function takes in a token list and adds the unique tokens to the vocabulary."""
    for tokens in  token_list:
        for token in tokens:
            if token in vocab_dict:
                vocab_dict[token] +=1
            else:
                vocab_dict[token] = 1
    return vocab_dict


def stoi(captions_list, chartoidx):
    """ Function to convert the string tokens to integers.
        Also adds the <start>, <end> & <unk> tokens to form the final list to be fed in the RNN
        INPUT --> list of captions with 
        OUPUT --> """
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


# Next we would add the training loop function which would run our training
# Now lets 
def training_loop(epochs, train_dataloader, val_dataloader, model, optimizer):
    """ Function to create the training loop which would take in the image and the caption train on it word by word """
    for epoch in range(epochs):
        loss_list = []
        for batch in tqdm(train_dataloader):
            image, caption, _ = batch
            # print(image.shape)
            # image = torch.unsqueeze(image, dim=0)
            # print(caption.shape)
            image = image.to(device = torch.device('cuda'))
            caption = caption.to(device = torch.device('cuda'))
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