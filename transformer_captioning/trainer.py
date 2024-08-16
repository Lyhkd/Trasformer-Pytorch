import numpy as np
from utils import  decode_captions

import torch
from tqdm import tqdm


class Trainer(object):

    def __init__(self, model, train_dataloader, val_dataloader, learning_rate = 0.001, num_epochs = 10, print_every = 10, verbose = True, device = 'cuda', model_path=None, load_model=False):
      
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.print_every = print_every
        self.verbose = verbose 
        self.loss_history = []
        self.val_loss_history = []
        self.device = device
        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.model_path = model_path
        if model_path is not None and load_model:
            self.load_model(model_path)

    def loss(self, predictions, labels):
        #TODO - Compute cross entropy loss between predictions and labels. 
        #Make sure to compute this loss only for indices where label is not the null token.
        #The loss should be averaged over batch and sequence dimensions. 
        mask = (labels != self.model._null)  # 假设 <NULL> 的索引为 0
        # 选择有效的预测和标签
        predictions = predictions[mask].to(self.device)
        labels = labels[mask].to(self.device)
        # 计算损失
        loss = torch.nn.functional.cross_entropy(predictions, labels)
        return loss
    
    def val(self):
        """
        Run validation to compute loss and BLEU-4 score.
        """
        self.model.eval()
        val_loss = 0
        num_batches = 0
        for batch in self.val_dataloader:
            features, captions = batch[0].to(self.device), batch[1].to(self.device)
            logits = self.model(features, captions[:, :-1])

            loss = self.loss(logits, captions[:, 1:])
            val_loss += loss.detach().cpu().numpy()
            num_batches += 1

        self.model.train()
        return val_loss/num_batches

    def train(self):
        """
        Run optimization to train the model.
        """
        for i in tqdm(range(self.num_epochs)):
            epoch_loss = 0
            num_batches = 0
            for batch in self.train_dataloader:
                features, captions = batch[0].to(self.device), batch[1].to(self.device)
                # print(f'in train, feature shape & caption shape {features.shape, captions.shape, features[0], captions[0]}')
                logits = self.model(features, captions[:, :-1])

                loss = self.loss(logits, captions[:, 1:])
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                epoch_loss += loss.detach().cpu().numpy()
                num_batches += 1
                
            self.loss_history.append(epoch_loss/num_batches)
            if self.verbose and (i +1) % self.print_every == 0:
                self.val_loss_history.append(self.val())
                print( "(epoch %d / %d) loss: %f" % (i+1 , self.num_epochs, self.loss_history[-1]))    
                
                
    def save_model(self, path=None):
        """
        Save the model to a file.
        """
        if path is None:
            path = self.model_path
        torch.save(self.model.state_dict(), path)
        print("Model saved to %s." % path)
    
    def load_model(self, path=None):
        """
        Load the model from a file.
        """
        if path is None:
            path = self.model_path
        self.model.load_state_dict(torch.load(path))
        print("Model loaded from %s." % path)