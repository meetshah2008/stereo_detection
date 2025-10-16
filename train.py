import paddle
from datasets.stereo_dataset import StereoDetectionDataset
from models.stereo_rtdetr import StereoRTDETR
from losses.stereo_loss import StereoDetectionLoss
from engine.trainer import StereoTrainer

def main():
    # Configuration
    data_root = 'path/to/your/data'  # Change this to your data path
    img_size = 640
    batch_size = 4
    num_epochs = 100
    num_classes = 3  # Change based on your classes
    learning_rate = 0.001
    
    # Create dataset
    dataset = StereoDetectionDataset(data_root, img_size=img_size, augment=True)
    
    # Split dataset (simple version - you might want proper train/val split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = paddle.io.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = paddle.io.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = paddle.io.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Create model
    model = StereoRTDETR(num_classes=num_classes)
    
    # Create loss and optimizer
    criterion = StereoDetectionLoss(num_classes=num_classes)
    optimizer = paddle.optimizer.Adam(
        learning_rate=learning_rate, 
        parameters=model.parameters()
    )
    
    # Create trainer
    trainer = StereoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        save_dir='output'
    )
    
    # Start training
    print("Starting training...")
    trainer.train(num_epochs)

if __name__ == '__main__':
    main()