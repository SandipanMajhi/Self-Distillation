import torch
import json
import argparse
from utils.dataset import *

if __name__ == "__main__":
    with open("utils/params.json") as fp:
        data = json.load(fp)

    cora_params = data["cora"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ = dataloader(param=cora_params)

