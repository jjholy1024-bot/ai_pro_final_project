import os
import json
import yaml
import argparse
import torch
from PIL import Image
from tqdm import tqdm

from models import get_model
from utils.transforms import get_transforms


def predict(config_path, checkpoint_path, output_dir=None):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    data_dir = config['dataset']['data_dir']
    num_classes = config['model'].get('num_classes')
    if num_classes is None:
        classes_path = os.path.join(data_dir, 'classes.json')
        with open(classes_path, 'r', encoding='utf-8') as f:
            num_classes = len(json.load(f))
    config['model']['num_classes'] = num_classes

    model = get_model(**config['model']).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    transform = get_transforms('KFood', is_train=False)

    test_list_path = os.path.join(data_dir, 'splits', 'test.txt')
    with open(test_list_path, 'r', encoding='utf-8') as f:
        test_files = [line.strip() for line in f if line.strip()]

    predictions = {}
    for rel_path in tqdm(test_files, desc='Predicting'):
        img_path = os.path.join(data_dir, rel_path) + '.jpg'
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            pred = torch.argmax(out, dim=1).item()
        predictions[os.path.basename(rel_path) + '.jpg'] = pred

    if output_dir is None:
        output_dir = os.path.join('./runs', config['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, 'test_predictions.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Predictions saved to: {out_path}")
    print(f"Total: {len(predictions)} test images")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    predict(args.config, args.checkpoint, args.output_dir)
