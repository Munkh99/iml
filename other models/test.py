import numpy as np
import requests
import json
from itertools import islice
from dataset import get_test_data
import torch
from pathlib import Path
import yaml
from model import VGG16
from model import ResNet50
from model import EfficientNet
from utils import seed_everything

def feature_extraction(config, query_loader, gallery_loader, model):
    feature_query = dict()
    feature_gallery = dict()

    with torch.no_grad():
        for data, names in query_loader:
            data = data.to(config["device"])
            output = model.inference(data).to(config["device"]).cpu().numpy()

            for i, j in zip(names, output):
                feature_query[i] = j
        for data, names in gallery_loader:
            data = data.to(config["device"])
            output = model.inference(data).to(config["device"]).cpu().numpy()
            for i, j in zip(names, output):
                feature_gallery[i] = j
    return feature_query, feature_gallery


def find_distance(array1, array2):
    return np.linalg.norm(array1 - array2)


def submit(results, url="https://competition-production.up.railway.app/results/"):
    res = json.dumps(results)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['results']}")
        return result
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")
        return None



def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))


def find_similarity(feature_query, feature_gallery, N):
    result = dict()
    for q_name, q_feature in feature_query.items():
        tmp = dict()
        for g_name, g_feature in feature_gallery.items():
            distance = find_distance(q_feature, g_feature)  # returns distance score
            tmp[g_name] = distance
        tmp = {k: v for k, v in sorted(tmp.items(), key=lambda item: item[1])}  # sort tmp by values
        result[q_name] = take(N, tmp.keys())
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Testing Parser")
    # Misc
    parser.add_argument("--model_name", choices=["VGG16", "ResNet50", "EfficientNet"], required=True, help="Name of the model used")
    parser.add_argument("--checkpoint_path", required=False, type=str, default="./checkpoints",
                        help="Path of the checkpoints to test.")
    # Dataset parameters
    opt = parser.parse_args()  # parse the arguments, this creates a dictionary name : value

    # === Seed the training
    seed_everything()

    # === Load the configuration file
    checkpoint_path = Path(opt.checkpoint_path) / opt.model_name
    config_path = checkpoint_path / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        print(f"\tConfiguration file loaded from: {config_path}")

    # Create the model & Load the weights
    if opt.model_name == "VGG16":
        model = VGG16(config, config["out_layer_size"])
    if opt.model_name == "ResNet50":
        model = ResNet50(config, config["out_layer_size"])
    if opt.model_name == "EfficientNet":
        model = EfficientNet(config, config["out_layer_size"])

    ckpt = torch.load(checkpoint_path / "best.pth", map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()
    model.to(config["device"])

    query_loader, gallery_loader = get_test_data(config)
    # Get feature extractions
    feature_query, feature_gallery = feature_extraction(config=config, query_loader=query_loader, gallery_loader=gallery_loader, model=model)
    # similarity for each query img to feature gallery images
    top_n = 10
    result = find_similarity(feature_query, feature_gallery, top_n)

    # preparation for submit
    mydata = dict()
    mydata['groupname'] = "Capybara"
    mydata["images"] = result
    with open('data.json', 'w') as f:
        json.dump(mydata, f)

    submit(mydata)
