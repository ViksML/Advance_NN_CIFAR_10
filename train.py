import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from model.model import CustomCNN
from utils.dataset import get_dataloaders
from utils.trainer import Trainer
from config.config import Config

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    device = get_device()
    
    # Initialize model
    model = CustomCNN().to(device)
    summary(model.to('cpu'), input_size=(3, 32, 32))
    model = model.to(device)

    # Get dataloaders
    train_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS
    )

    # Setup training
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        betas=(0.9, 0.999), 
        eps=1e-08
    )
    
    # Using ReduceLROnPlateau with updated parameters
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',         # Reduce LR when metric stops decreasing
        factor=0.1,         # Reduce LR by a factor of 10
        patience=5,         # Number of epochs with no improvement after which LR will be reduced
        min_lr=1e-6         # Minimum LR
    )

    # Using OneCycleLR scheduler
    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=Config.MAX_LR,
    #     epochs=Config.EPOCHS,
    #     steps_per_epoch=len(train_loader),
    #     pct_start=Config.PCT_START,
    #     div_factor=Config.DIV_FACTOR,
    #     final_div_factor=Config.FINAL_DIV_FACTOR
    # )

    # Initialize trainer
    trainer = Trainer(model, criterion, optimizer, scheduler, device)
    
    # Training loop
    for epoch in range(1, Config.EPOCHS + 1):
        print(f'Epoch: {epoch}')
        
        train_acc = trainer.train_epoch(train_loader)
        test_acc = trainer.test_epoch(test_loader)
        
        if test_acc > trainer.best_acc:
            trainer.best_acc = test_acc
            trainer.best_epoch = epoch
            trainer.save_checkpoint(epoch, test_acc)

        if test_acc >= Config.TARGET_ACCURACY:
            print(f"Reached target accuracy of {Config.TARGET_ACCURACY*100}% at epoch {epoch}")
            break

    print(f'Best test Accuracy: {trainer.best_acc * 100:.2f}%, Epoch: {trainer.best_epoch}')

if __name__ == "__main__":
    main() 