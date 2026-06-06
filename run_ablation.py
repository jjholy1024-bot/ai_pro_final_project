import os
import yaml
import subprocess

configs = [
    {"use_se": False, "ensemble_count": 1, "name": "Fast_Base_128", "bs": 128},
    {"use_se": True, "ensemble_count": 1, "name": "Fast_SE_128", "bs": 128},
    {"use_se": False, "ensemble_count": 2, "name": "Fast_Ens_128", "bs": 64},
]

template = """
experiment_name: "{name}"
dataset:
  name: "KFood"
  data_dir: "./data/kfood_dataset"
  batch_size: {bs}
  num_workers: 4
model:
  name: "MyModel"
  in_channels: 3
  dropout: 0.5
  use_se: {use_se}
  ensemble_count: {ensemble_count}
train:
  epochs: 5
  learning_rate: 0.001
"""

for cfg in configs:
    cfg_path = f"configs/{cfg['name']}.yaml"
    with open(cfg_path, 'w') as f:
        f.write(template.format(**cfg))
    
    print(f"--- Running {cfg['name']} ---")
    subprocess.run(["python", "train.py", "--config", cfg_path])
