import json
import argparse
from utils import dataset

if __name__ == "__main__":
    with open("utils/params.json") as fp:
        data = json.load(fp)

    print(data)

