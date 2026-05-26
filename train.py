import os
import csv
import shutil
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from datasets import get_dataset, get_dataloader
from models import get_model


def main(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"Starting Experiment: {config['experiment_name']}")

    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    exp_name = config['experiment_name']
    save_dir = os.path.join("./runs", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    shutil.copy(config_path, os.path.join(save_dir, "config_backup.yaml"))

    log_path = os.path.join(save_dir, "train_log.csv")
    log_file = open(log_path, 'w', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'train_loss', 'val_acc'])

    print("Loading Datasets and DataLoaders...")

    train_dataset = get_dataset(is_train=True, **config['dataset'])
    train_loader  = get_dataloader(dataset=train_dataset, is_train=True, **config['dataset'])

    val_dataset = get_dataset(is_train=False, **config['dataset'])
    val_loader  = get_dataloader(dataset=val_dataset, is_train=False, **config['dataset'])

    num_classes = len(train_dataset.classes)
    config['model']['num_classes'] = num_classes
    model = get_model(**config['model']).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    epochs = config['train']['epochs']

    best_acc = 0.0

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0

            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
            for images, labels in train_pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_train_loss = running_loss / len(train_loader)

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
                for images, labels in val_pbar:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = 100 * correct / total
            print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f} | Val Acc={val_acc:.2f}%")

            log_writer.writerow([epoch, f"{avg_train_loss:.4f}", f"{val_acc:.2f}"])
            log_file.flush()

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                print(f"  -> Best model saved (Val Acc: {val_acc:.2f}%)")
    finally:
        log_file.close()

    print("Training Complete.")
    print(f"Best Val Acc: {best_acc:.2f}%")
    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)
