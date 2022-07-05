import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
import pytorch_lightning as pl
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

wandb.login(key='028f2bb73f8d6aa6ae54ccfb3d22b397309bbdd0')

class Shell(pl.LightningModule):
    def __init__(self, model, CONFIG, device='cuda', passport=None, wandb=True):
        super().__init__()
        self.model = model
        self.CONFIG = CONFIG
        self.device = device
        self.scheduler = CONFIG.get('scheduler', None)
        self.wandb=wandb
        
        """Set logger"""
        if passport:
                logger = pl.loggers.wandb.WandbLogger(
                    name=passport["exp_name"], 
                    project=passport.get("project_name", 'man_Project'),
                    tags=passport.get("tags", ['tag0']),
                    group=passport.get("group", "group0"))
            
                logger.log_hyperparams(self.CONFIG)
                logger.watch(self.model, log='all', log_freq=1)
        else:
                logger = None
                
        self.to(device)
        self.model.to(device)
        
    def forward(self, inp_features):
        return self.model(inp_features)
    
    def step(self, batch, mode='train'):
        pass
    
    def training_step(self, batch, batch_idx):
        loss = self.step(batch, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
            self.step(batch, 'val')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.CONFIG['learning_rate'])
        
        if self.scheduler == 'linear':
            lambda1 = lambda x: (1 - 0.9*x/100)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
            return {'optimizer' : optimizer, 'scheduler' : scheduler}
        return optimizer
    
class LightningShell(Shell):
    def __init__(self, model, CONFIG, device='cuda', passport=None):
        super().__init__(model, CONFIG, device, passport)
        
    def step(self, batch, mode='train'):
        assert mode in ['train', 'val', 'test']
        inputs, mask, labels, labels_for_loss = batch
        inputs = inputs.to(self.device)
        mask = mask.to(self.device)
        labels = labels.to(self.device)
            
        out = model(inputs, mask, labels)
        logits = out.logits
        loss = out.loss
            
        #Logging
        probabilities = nn.Softmax(logits.detach(), dim=0)
        predictions = torch.argmax(probabilities, dim=0).view(-1).numpy()
            
        labels_np = labels.detach().cpu().view(-1).numpy()
            
        accuracy = accuracy_score (labels_np, predictions)
        p_scr    = precision_score(labels_np, predictions)
        r_scr    = recall_score   (labels_np, predictions)
        f_scr    = f1_score       (labels_np, predictions)
            
        H = np.histogram(prediction)
        H1 = np.histogram(probabilities.detach().cpu().numpy()[:,0])
        
        if self.wandb:
            WandB = self.logger.experiment
            if mode == 'train':
                WandB.log({'train/loss' : loss.item(), 
                           'train/accuracy': accuracy,
                           'train/precision': p_scr,
                           'train/recall': r_scr,
                           'train/F1': f_scr,
                           'train/preds': wandb.Histogram(np_histogram = H),
                           'train/distribution_p' : wandb.Histogram(np_histogram = H1)})
            else:
                WandB.log({'val/loss' : loss.item(), 
                           'val/accuracy' : accuracy,
                           'val/precision': p_scr,
                           'val/recall': r_scr,
                           'val/F1': f_scr,
                           'val/preds': wandb.Histogram(np_histogram = H),
                           'val/distribution_p' : wandb.Histogram(np_histogram = H1)})
        if mode == 'train':
            return loss
