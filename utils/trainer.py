import torch
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_acc = 0
        self.best_epoch = 0

    def train_epoch(self, train_loader):
        self.model.train()
        pbar = tqdm(train_loader)
        train_loss = 0
        correct = 0
        processed = 0
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            train_loss += loss.item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            
            loss.backward()
            self.optimizer.step()
            
            pbar.set_description(desc= f'Training: Loss={loss.item():.4f} Batch={batch_idx} Accuracy={100*correct/processed:.2f}%')
        
        train_loss /= len(train_loader.dataset)
        accuracy = 100. * correct / len(train_loader.dataset)
        print(f'Training: Average loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)')
        
        return train_loss

    def test_epoch(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        # Print metrics including current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f'Testing: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
        #print(f'Learning Rate: {current_lr:.6f}\n')
        
        # Step scheduler with test loss
        self.scheduler.step(test_loss)
        
        return correct / len(test_loader.dataset)

    def save_checkpoint(self, epoch, test_acc):
        os.makedirs('model', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'test_acc': test_acc,
        }, 'model/best_model.pth') 