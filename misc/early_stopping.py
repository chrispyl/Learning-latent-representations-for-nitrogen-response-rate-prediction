import numpy as np
import torch

class EarlyStopping:
    '''Early stops the training if validation loss doesn't improve after a given patience
    
    Attributes
    ----------
    patience : int
        How long to wait after last time validation loss improved
    verbose : torch.Tensor
        If True, prints a message for each validation loss improvement
    delta : float
        Minimum change in the monitored quantity to qualify as an improvement
    path : str
        Path for the checkpoint to be saved to
    trace_func : fn
        trace print function
    '''
    
    def __init__(self, patience=7, verbose=False, delta=0, save_path='model_checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model, epoch, total_epochs):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.trace_func(f'Epoch {epoch}/{total_epochs}, Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'Epoch {epoch}/{total_epochs}, Validation loss increased, EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.trace_func(f'Epoch {epoch}/{total_epochs}, Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            self.save_checkpoint(val_loss, model)
            

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        if self.verbose:
            self.trace_func(f'Model saved')
        self.val_loss_min = val_loss