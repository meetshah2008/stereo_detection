import os
import time
import paddle
from tqdm import tqdm

class StereoTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, save_dir='output'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
    
    def train(self, epochs):
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch_idx, batch in enumerate(progress_bar):
                left_imgs = batch['left_image']
                right_imgs = batch['right_image']
                targets = {
                    'bboxes': batch['bboxes'],
                    'classes': batch['classes'],
                    'disparities': batch['disparities']
                }
                
                # Forward pass
                outputs = self.model(left_imgs, right_imgs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(self.train_loader)
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, avg_loss)
            
            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f'Validation Loss: {val_loss:.4f}')
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with paddle.no_grad():
            for batch in self.val_loader:
                left_imgs = batch['left_image']
                right_imgs = batch['right_image']
                targets = {
                    'bboxes': batch['bboxes'],
                    'classes': batch['classes'],
                    'disparities': batch['disparities']
                }
                
                outputs = self.model(left_imgs, right_imgs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        
        path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pdparams')
        paddle.save(checkpoint, path)
        print(f'Checkpoint saved: {path}')