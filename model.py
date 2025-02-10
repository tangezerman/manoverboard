from ultralytics import YOLO
import yaml
from pathlib import Path
import multiprocessing


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
        epochs=100,
        imgsz=640,
        batch=50,
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

    # Validate the model
    results = model.val()

    # Optional: Run inference on test set
    results = model.predict(source='dataset/test/images', save=True, conf=0.25)


if __name__ == '__main__':
    multiprocessing.freeze_support()  # Add this line for Windows support
    main()
