import torch
import torch.nn as nn
torch.random.manual_seed(42)
torch.cuda.manual_seed(42)

class CustomResNet(nn.Module):
    def __init__(self, chartoidx, idxtochar, pretrained_model = None ):
        super().__init__()



        for parameter in pretrained_model.parameters():
            parameter.requires_grad = False

        
        pretrained_model.fc = nn.Linear(2048,256)

        self.chartoidx = chartoidx
        self.idxtochar = idxtochar
        self.backbone = nn.Sequential(pretrained_model)
        # Till the previous step we have a pretrained Resnet whose weights we have frozen. Now we will define the RNN

        # self.embedding_matrix = torch.randn(len(chartoidx), 50, requires_grad=True) / (len(chartoidx)**0.5)
        self.embedding_matrix = nn.Embedding(len(self.chartoidx), 10)
        self.rnn = nn.RNNCell(input_size=10, hidden_size = 256, nonlinearity='tanh')
        self.output_layer = nn.Linear(256,len(self.chartoidx))

        self.cross_entopy_loss = nn.CrossEntropyLoss()

    def forward(self, image, caption_list):
        out_backbone = self.backbone(image)
        print(f"The output backbone image shape is {out_backbone.shape}")
        hidden = out_backbone.squeeze(0)
        # print(f"the hidden shape from image is {hidden.shape}")
        i = 0
        # new_char_id = char_id
        loss_word_list = []
        # current_word_id = caption_list[i]
        # target_index = caption_list[i+1]
        # so what we can do is get the embedding matrix of the captions beforehand from the embedding matrix by passing a list
        # Then what we can do is that we can loop through each of the embedding and calculate the rnn-cell output
        caption_embeddings = self.embedding_matrix(caption_list)
        while True:
            
            if i == 0:
                current_word_id = caption_list[0]    # start at the first word i.e the <start> token
                target_index = caption_list[i+1]
            if self.idxtochar[target_index] == '<end>':
                break
            if i!=0 and i<len(caption_list)-1:
                target_index = caption_list[i+1]
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
            current_word_loss = self.cross_entopy_loss(probs, target_index)
            # append the current_word_loss to loss_word_list
            loss_word_list.append(current_word_loss)
            
            # Second we would use the index of the current most probable output word as an input of the next time-step
            current_word_id = next_word_id
            # increase the value of i by 1 so that we can get the target word for the next input word
            i+=1
            
        final_loss = torch.sum(torch.tensor(loss_word_list, requires_grad=True))
        return final_loss