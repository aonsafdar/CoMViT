
import subprocess

subprocess.run([
    "python", "/mnt/storage/aonsafdar/cct/Compact-Transformers/train.py", "./data",
    "--dataset", "medmnist",
    "--train-split", "pneumoniamnist",
    "--model", "cct_7_7x2_224",
    "--input-size", "3", "224", "224",
    "--num-classes", "2",
    "--batch-size", "512",
    "--epochs", "300",
    "--sched", "cosine",
    "--opt", "adamw",
    "--lr", "0.0005",
    "--drop", "0.1",
    "--drop-path", "0.1",
    "--clip-grad", "1.0",
    "--mean", "0.5", "0.5", "0.5",
    "--std", "0.5", "0.5", "0.5",
    "--amp",
    "--log-wandb",
    "--experiment", "pneumoniamnist",
    "--output", "./output/pneumoniamnist"
])
