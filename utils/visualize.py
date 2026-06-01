import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_learning_curves(log_path, save_path):
    df = pd.read_csv(log_path)
    
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.title('Training Loss Trend')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['val_acc'], label='Val Acc', color='orange')
    plt.title('Validation Accuracy Trend')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Learning curves saved to: {save_path}")

def plot_confusion_matrix(results_path, save_path):
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    cm = results['confusion_matrix']
    classes = results['class_order']
    
    # Too many classes might make the plot unreadable, 
    # but let's try a large figure or focus on top/bottom
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, xticklabels=classes, yticklabels=classes, annot=False, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix heatmap saved to: {save_path}")

def plot_per_class_accuracy(results_path, save_path, top_n=20):
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    per_class = results['per_class_accuracy']
    # Sort and take bottom N
    sorted_acc = sorted(per_class.items(), key=lambda x: x[1])
    bottom_n = sorted_acc[:top_n]
    
    names, values = zip(*bottom_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(names, values, color='salmon')
    plt.title(f'Bottom {top_n} Classes by Accuracy')
    plt.xlabel('Accuracy (%)')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Per-class accuracy plot saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True, help="Path to the experiment directory in /runs")
    args = parser.parse_args()
    
    log_csv = os.path.join(args.exp_dir, "train_log.csv")
    val_json = os.path.join(args.exp_dir, "val_results.json")
    
    if os.path.exists(log_csv):
        plot_learning_curves(log_csv, os.path.join(args.exp_dir, "learning_curves.png"))
    
    if os.path.exists(val_json):
        plot_confusion_matrix(val_json, os.path.join(args.exp_dir, "confusion_matrix.png"))
        plot_per_class_accuracy(val_json, os.path.join(args.exp_dir, "bottom_accuracy.png"))
