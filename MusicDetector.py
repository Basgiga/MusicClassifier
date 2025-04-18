import os
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

# PARAMETERS
SEED = 42
BATCH_SIZE = 32
hidden_units = 40
learning_rate = 0.001
EPOCHS = 30
DROPOUT_RATE = 0.2

#DIRS
CSV_FILE_DIR = "/Users/bartoszborkowski/Downloads/Data/features_30_sec.csv"
IMG_DIR = "/Users/bartoszborkowski/Downloads/Data/images_original/"

class MusicGenreIMAGEDataset(Dataset):
    def __init__(self, img_dir,csv_dir, labels_df= None, transform = None):
        self.img_dir = img_dir
        self.csv_data = pd.read_csv(csv_dir)
        self.csv_features = []
        self.csv_labels = []
        self.genres = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
        self.file_paths = []
        self.labels = []
        for genre in self.genres:
            genre_path = os.path.join(img_dir, genre)
            for file_name in os.listdir(genre_path):
                if file_name.endswith('.png'):
                    self.file_paths.append(os.path.join(genre_path, file_name))
                    self.labels.append(genre)
        for i in range(len(self.csv_data)):
            which_column = 0
            feature = []
            for row in self.csv_data:
                if which_column == 0:
                    pass
                elif which_column == 59:
                    self.csv_labels.append(self.csv_data[row][i])
                else:
                    feature.append(self.csv_data[row][i])
                which_column+=1
            self.csv_features.append(feature)
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.encoded_labels[idx]
        csv_features = torch.tensor(self.csv_features[idx], dtype=torch.float32)  # convert to tensor
        if self.transform:
            image = self.transform(image)
        return image,csv_features, torch.tensor(label, dtype=torch.long)

class SigmoidAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SigmoidAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SpectoCNN(nn.Module):
    def __init__(self, num_classes,csv_features_dim):
        super(SpectoCNN,self).__init__()
        self.Conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels=hidden_units, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride = 2)
        )
        self.Conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units, out_channels=hidden_units*2, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(hidden_units*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride = 2)
        )
        self.Conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units*2, out_channels=hidden_units*4, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(hidden_units*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride = 2)
        )

        self.Sigblock_1 = SigmoidAttentionBlock(hidden_units*4)

        self.Conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units*4, out_channels=hidden_units*6, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(hidden_units*6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride = 2)
        )
        self.Conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units*6, out_channels=hidden_units*8, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(hidden_units*8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride = 2)
        )

        self.Sigblock_2 = SigmoidAttentionBlock(hidden_units*8)


        self.transfer = nn.Sequential(
            nn.Linear(hidden_units*8*8*8,512),
            nn.ReLU(),
            nn.Dropout(p = DROPOUT_RATE)
        )

        #MLP for the csv features
        self.mlp = nn.Sequential(
            nn.Linear(csv_features_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, num_classes)
        )



        self.Classifier = nn.Sequential(
            nn.Linear(512 + num_classes, 256),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(256, num_classes)
        )


    def forward(self, x, csv):

        # spectogram cnn result
        x = self.Conv_block_1(x)
        x = self.Conv_block_2(x)
        x = self.Conv_block_3(x)
        x = self.Sigblock_1(x)
        x = self.Conv_block_4(x)
        x = self.Conv_block_5(x)
        x = self.Sigblock_2(x)
        x = x.view(x.size(0), -1)
        x_cnn = self.transfer(x)

        # csv features mlp result
        x_csv = self.mlp(csv)


        combinedfeat = torch.cat((x_cnn,x_csv), dim = 1)

        output_comb =  self.Classifier(combinedfeat)

        return output_comb

def Load_data(IMG_DIR,CSV_FILE_DIR, IMG_TRANSFORM):
    full_dataset= MusicGenreIMAGEDataset(img_dir=IMG_DIR,csv_dir=CSV_FILE_DIR, transform = IMG_TRANSFORM)

    csv_data = pd.read_csv(CSV_FILE_DIR)
    csv_f_dim = csv_data.shape[1] - 2

    indices = list(range(len(full_dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=0.2,
                                                   random_state=SEED)
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(full_dataset.label_encoder.classes_)
    print(f"Liczba gatunkow muzycznych: {num_classes}")
    print(f"rozmiar zbioru treningowego: {len(train_dataset)}")

    return num_classes, train_loader, test_loader, csv_f_dim

def Load_model(num_class, csv_f_dim):
    model = SpectoCNN(num_classes = num_class, csv_features_dim=csv_f_dim)
    return model

def Train_model(model, loss_fn, optimizer,traindataloader, epochs = EPOCHS):

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n----")
        train_loss = 0
        for batch, (image, csv_features, label) in enumerate(traindataloader):
            model.train()
            image, csv_features, label = image.to(device), csv_features.to(device), label.to(device)

            # forward pass
            y_pred = model(image, csv_features)

            # loss
            loss = loss_fn(y_pred,label)
            train_loss += loss.item()

            # back
            optimizer.zero_grad()
            loss.backward()

            #optimizer
            optimizer.step()
        train_loss/= len(traindataloader)
    return model

def Evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, csv_features, labels in testloader:
            inputs,csv_features, labels = inputs.to(device), csv_features.to(device), labels.to(device)
            outputs = model(inputs, csv_features)

            # get the top probabilty label
            _, predicted = torch.max(outputs.data,1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy = {accuracy}")

    return model


def Main():

    # image transforms before getting into the model
    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),  # resize
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # normalize the pixels for each channel (RGB)
    ])

    #load specto data
    num_classes, train_loader, test_loader, csv_f_dim = Load_data(IMG_DIR, CSV_FILE_DIR, image_transforms)

    #load model
    model = Load_model(num_classes, csv_f_dim= csv_f_dim)
    #print(model)

    #loss function will be CrossEntropy
    loss_fn = nn.CrossEntropyLoss()

    # optimizer will be ADAM
    optimizer = optim.Adam(model.parameters(), lr= learning_rate)

    #lets get ready for training!
    loss_fn.to(device)
    model.to(device)

    model = Train_model(model, loss_fn, optimizer, train_loader)

    # now lets evaluate the model
    model = Evaluate_model(model,test_loader)

Main()