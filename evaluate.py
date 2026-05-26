import os
import json
import yaml
import argparse
import torch
from tqdm import tqdm

from datasets import get_dataset, get_dataloader
from models import get_model


def evaluate(config_path, checkpoint_path, output_dir=None):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    val_dataset = get_dataset(is_train=False, split='val', **config['dataset'])
    val_loader  = get_dataloader(dataset=val_dataset, is_train=False, **config['dataset'])

    num_classes = len(val_dataset.classes)
    config['model']['num_classes'] = num_classes
    model = get_model(**config['model']).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    idx_to_class = {v: k for k, v in val_dataset.class_to_idx.items()}
    confusion = [[0] * num_classes for _ in range(num_classes)]
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluating (val)'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                confusion[t][p] += 1

    total   = len(all_labels)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    overall_acc = 100.0 * correct / total

    per_class_acc = {}
    for cls_idx in range(num_classes):
        cls_name    = idx_to_class[cls_idx]
        cls_total   = sum(confusion[cls_idx])
        cls_correct = confusion[cls_idx][cls_idx]
        per_class_acc[cls_name] = round(100.0 * cls_correct / cls_total, 2) if cls_total > 0 else 0.0

    results = {
        "experiment_name":   config['experiment_name'],
        "checkpoint":        checkpoint_path,
        "split":             "val",
        "overall_accuracy":  round(overall_acc, 2),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix":  confusion,
        "class_order":       [idx_to_class[i] for i in range(num_classes)],
    }

    if output_dir is None:
        output_dir = os.path.join('./runs', config['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)

    result_path = os.path.join(output_dir, 'val_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nVal Accuracy: {overall_acc:.2f}%")
    print("Per-class accuracy (bottom 10):")
    for cls_name, acc in sorted(per_class_acc.items(), key=lambda x: x[1])[:10]:
        print(f"  {cls_name}: {acc:.2f}%")
    print(f"\nResults saved to: {result_path}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.output_dir)
