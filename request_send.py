import glob
import json
import os
from itertools import islice
import requests
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np
import network


def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))


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


class competitionSet(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.root = path
        self.image_paths = glob.glob(self.root + "*.*")

        self.transform = transforms.Compose([
            transforms.Resize(size=(224, 224), antialias=True),
            # transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")  # convert rgb images to grayscale
        img_name = os.path.basename(self.image_paths[idx])
        img = self.transform(img)
        img = img.unsqueeze(0)
        return img, img_name


def get_query_and_gallery(path_to_query,
                          path_to_gallery):
    query_set = competitionSet(path_to_query)
    gallery_set = competitionSet(path_to_gallery)
    return query_set, gallery_set


def distance_estimator_2(query_set, gallery_set, model, DEVICE, N):
    model.eval()
    result = dict()
    g = {}
    q = {}
    print("--------------------------------------------------------------------")
    with torch.no_grad():
        for img, img_name in query_set:
            img = img.to(DEVICE)
            output = model(img)
            q[img_name] = output

        for img_g, img_name_g in gallery_set:
            img_g = img_g.to(DEVICE)
            output = model(img_g)
            g[img_name_g] = output

    print(len(q))
    print(len(g))

    print("--------------------------------------------------------------------")
    tmp = []
    for idx, (img1, out1) in enumerate(q.items()):
        print(idx, "-------------------------")
        for img2, out2 in g.items():
            distance = torch.norm(out1 - out2, dim=1, p=2)
            tmp.append((img2, np.round(distance.item(), 6)))  # Append tuple of (img2, dissim) to the list
        tmp = sorted(tmp, key=lambda x: x[1], reverse=False)  # Sort tmp by distance values

        tt = take(N, [k for k, _ in tmp])  # Extract top N img2 values from tmp
        result[img1] = tt  # Extract top N img2 values from tmp
        tmp = []  # Clear the list for the next iteration
    return result


def main():
    DEVICE = torch.device("cpu") # if it uses cpu
    # DEVICE = torch.device("mps") # if it uses gpu on mac
    # DEVICE = torch.device("cuda") # if it uses gpu

    model = network.TripletNetwork().to(DEVICE)

    checkpoint_file = '/Users/munkhdelger/PycharmProjects/ML_competition/checkpoints/Triplet/best.pth'

    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)

    ### Find distance between 2 images ###

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the images to the input size of ResNet
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
    ])

    # Load and preprocess the input images
    image1 = Image.open("/Users/munkhdelger/PycharmProjects/ML_competition/data/test_data/query_cropped/17d30301d158d0912af8b768cc125710bda03a40.jpg").convert("RGB")
    image2 = Image.open("/Users/munkhdelger/PycharmProjects/ML_competition/data/test_data/gallery_cropped/c67912876ef1ee503957abd9ae87721cb883f864.jpg").convert("RGB")
    input1 = preprocess(image1).unsqueeze(0)  # Add a batch dimension
    input2 = preprocess(image2).unsqueeze(0)  # Add a batch dimension

    input1 = input1.to(DEVICE)
    input2 = input2.to(DEVICE)

    # Pass the inputs through the ResNet model
    output1 = model(input1)
    output2 = model(input2)

    # Compute the Euclidean distance between the outputs
    distance = torch.norm(output1 - output2, p=2)
    print('Distance: {:.4f}'.format(distance.item()))


    #----------------------------------------------
    # For the competition

    # path_to_query = '/Users/munkhdelger/PycharmProjects/ML_competition/data/test_data/query_cropped/'
    # path_to_gallery = '/Users/munkhdelger/PycharmProjects/ML_competition/data/test_data/gallery_cropped/'
    #
    # query_set, gallery_set = get_query_and_gallery(path_to_query, path_to_gallery)
    # result = distance_estimator_2(query_set, gallery_set, model, DEVICE, N=10)
    #
    # query_random_guess = dict()
    # query_random_guess['groupname'] = "Capybara"
    # query_random_guess["images"] = result
    # with open('/Users/munkhdelger/PycharmProjects/ML_competition/Metric learning/data.json', 'w') as f:
    #     json.dump(query_random_guess, f)
    #
    # # Opening JSON file
    # f = open('/Users/munkhdelger/PycharmProjects/ML_competition/Metric learning/data.json', 'r')
    # data = json.load(f)
    # submit(data)


if __name__ == '__main__':
    main()
