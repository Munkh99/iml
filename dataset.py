import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import random


class TinyImageNetDataset_Triplet():
    def __init__(self, df, path, transform=None):
        self.data_csv = df
        self.transform = transform
        self.path = path

        self.images = df.iloc[:, 0].values
        self.labels = df.iloc[:, 1].values
        self.index = df.index.values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        anchor_image_name = self.images[item]
        anchor_image_path = self.path + '/' + anchor_image_name
        ###### Anchor Image #######
        anchor_img = Image.open(anchor_image_path).convert('RGB')

        anchor_label = self.labels[item]
        positive_indexes = np.where((self.labels == anchor_label) & (self.index != item))[0]
        positive_item = random.choice(positive_indexes)
        positive_image_name = self.images[positive_item]
        positive_image_path = self.path + '/' + positive_image_name
        positive_img = Image.open(positive_image_path).convert('RGB')
        negative_indexes = np.where((self.labels != anchor_label) & (self.index != item))[0]
        negative_item = random.choice(negative_indexes)
        negative_image_name = self.images[negative_item]
        negative_image_path = self.path + '/' + negative_image_name
        negative_img = Image.open(negative_image_path).convert('RGB')

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        return anchor_img, positive_img, negative_img, anchor_label


class TinyImageNetDataset_Triplet_haar():
    def __init__(self, df, path, transform=None):
        self.data_csv = df
        self.transform = transform
        self.path = path
        self.face_detector = cv2.CascadeClassifier(
            '/Users/munkhdelger/PycharmProjects/ML_competition/haarcascade_frontalface_default.xml')


        self.images = df.iloc[:, 0].values
        self.labels = df.iloc[:, 1].values
        self.index = df.index.values

    def __len__(self):
        return len(self.images)

    def crop_image(self, image_path, crop_size=224):
        # Load the image
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(130, 130))
        # print(image_path, len(faces))
        # Check if faces are detected
        if len(faces) > 0:
            # Find the largest face
            largest_face_index = 0
            largest_face_area = 0
            for i, (x, y, w, h) in enumerate(faces):
                area = w * h
                if area > largest_face_area:
                    largest_face_area = area
                    largest_face_index = i

            # Crop the largest face
            (x, y, w, h) = faces[largest_face_index]
            cropped_image = image[y:y + h, x:x + w]

            return cropped_image
        else:
            # print('No face detected in the image.')
            return image

    def __getitem__(self, item):
        anchor_image_name = self.images[item]
        anchor_image_path = self.path + '/' + anchor_image_name

        # Crop anchor image
        anchor_image = self.crop_image(anchor_image_path)
        anchor_img = Image.fromarray(anchor_image)
        anchor_label = self.labels[item]

        positive_indexes = np.where((self.labels == anchor_label) & (self.index != item))[0]
        positive_item = random.choice(positive_indexes)
        positive_image_name = self.images[positive_item]
        positive_image_path = self.path + '/' + positive_image_name

        # Crop positive image
        positive_image = self.crop_image(positive_image_path)
        positive_img = Image.fromarray(positive_image)

        negative_indexes = np.where((self.labels != anchor_label) & (self.index != item))[0]
        negative_item = random.choice(negative_indexes)
        negative_image_name = self.images[negative_item]
        negative_image_path = self.path + '/' + negative_image_name

        # Crop negative image
        negative_image = self.crop_image(negative_image_path)
        negative_img = Image.fromarray(negative_image)

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        return anchor_img, positive_img, negative_img, anchor_label


def get_dataset_triplet(config, test_size=0.2, IMAGE_SIZE=224):
    transform = transforms.Compose([
        transforms.CenterCrop(size=(IMAGE_SIZE, IMAGE_SIZE)),
        # transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    data_csv = pd.read_csv(config["data_csv"], delimiter=" ")
    data_root = config["data_root"]
    # Split the data into train and test sets
    train_data, test_data = train_test_split(data_csv, test_size=test_size, random_state=42)

    train_dataset = TinyImageNetDataset_Triplet(train_data, path=data_root,
                                                transform=transform)
    test_dataset = TinyImageNetDataset_Triplet(test_data, path=data_root,
                                               transform=transform)

    train_dl = DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=True, num_workers=4,
                          pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=config["val_batch_size"], shuffle=False, num_workers=4,
                         pin_memory=True)

    return train_dl, test_dl
