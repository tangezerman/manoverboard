from ultralytics import YOLO
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_training_metrics(results_csv_path):
    """
    Plot training metrics from YOLO training results CSV
    Args:
        results_csv_path: Path to the results.csv file generated during training
    """
    # Set the style
    plt.style.use('bmh')
    sns.set_palette("husl")
    
    # Read the CSV file
    df = pd.read_csv(results_csv_path)
    
    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('YOLO Training Metrics', fontsize=16, y=0.95)
    
    # Plot training losses
    train_losses = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
    for loss in train_losses:
        ax1.plot(df['epoch'], df[loss], label=loss.split('/')[-1].replace('_', ' ').title(), marker='o', markersize=2)
    ax1.set_title('Training Losses vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss Value')
    ax1.legend()
    ax1.grid(True)
    
    # Plot validation losses
    val_losses = ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']
    for loss in val_losses:
        ax2.plot(df['epoch'], df[loss], label=loss.split('/')[-1].replace('_', ' ').title(), marker='o', markersize=2)
    ax2.set_title('Validation Losses vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Value')
    ax2.legend()
    ax2.grid(True)
    
    # Plot mAP metrics
    metrics = ['metrics/precision(B)', 'metrics/recall(B)', 
              'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    labels = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
    
    for metric, label in zip(metrics, labels):
        ax3.plot(df['epoch'], df[metric], label=label, marker='o', markersize=2)
    ax3.set_title('Performance Metrics vs Epoch')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Metric Value')
    ax3.legend()
    ax3.grid(True)
    
    # Plot learning rates
    lrs = ['lr/pg0', 'lr/pg1', 'lr/pg2']
    for lr in lrs:
        if lr in df.columns:
            ax4.plot(df['epoch'], df[lr], label=f'LR {lr.split("/")[-1]}', marker='o', markersize=2)
    ax4.set_title('Learning Rates vs Epoch')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.legend()
    ax4.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(results_csv_path).parent / 'training_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    # Display plot
    plt.show()


def main():
    # Define the classes based on your list
    classes = ['cross over', 'drill through', 'one foot above second',
               'sit', 'two feet off the ground', 'two feet on first']

    # Create YAML configuration file
    data_yaml = {
        'path': str(Path('.').absolute()),  # Path to the root directory
        'train': 'dataset/train/images',    # Train images path
        'val': 'dataset/val/images',        # Validation images path
        'test': 'dataset/test/images',      # Test images path

        'names': {i: name for i, name in enumerate(classes)},  # Class names
        'nc': len(classes)  # Number of classes
    }

    # Save the YAML configuration
    with open('dataset.yaml', 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    # Initialize YOLO model
    model = YOLO('yolo11n.pt')

    # Train the model
    results = model.train(
        data='dataset.yaml',
        epochs=65,
        imgsz=640,
        batch=100,
        device='0',
        workers=8,
        patience=20,
        project='runs',
        name='yolo_training',
        pretrained=True,
        optimizer='Adam',
        lr0=0.001,
        cos_lr=True,
        save=True,
        save_period=10,
        exist_ok=True,
        verbose=True
    )

    # Plot training metrics
    results_csv = Path('runs/yolo_training/results.csv')
    if results_csv.exists():
        plot_training_metrics(results_csv)
    else:
        print(f"Results file not found at {results_csv}")

    # Validate the model
    results = model.val()

    # Optional: Run inference on test set
    results = model.predict(source='dataset/test/images', save=True, conf=0.25)


if __name__ == '__main__':
    main()