# filepath: /home/dekelshoot/Bureau/faster_rcnn/src/training_model.py
"""# Importing libraries"""

import os
import numpy as np
import torch
from sklearn import preprocessing
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import draw_bounding_boxes
from PIL import Image
from torchvision.ops import box_iou
from engine import train_one_epoch
from utils import MetricLogger, SmoothedValue

"""# Preprocessing"""

BASE_PATH = "./pascal-voc-2012/VOC2012"
XML_PATH = os.path.join(BASE_PATH, "Annotations")
IMG_PATH = os.path.join(BASE_PATH, "JPEGImages")
XML_FILES = [os.path.join(XML_PATH, f) for f in os.listdir(XML_PATH)]

class XmlParser(object):
    # Implementation of XML parsing logic goes here
    pass

def xml_files_to_df(xml_files):
    # Implementation of XML to DataFrame conversion goes here
    pass

df = xml_files_to_df(XML_FILES)
df.head()

# check values for per class
df['names'].value_counts()

# remove .jpg extension from image_id
df['img_id'] = df['image_id'].apply(lambda x:x.split('.')).map(lambda x:x[0])
df.drop(columns=['image_id'], inplace=True)
df.head()

dataframe = df.copy()

# classes need to be in int form so we use LabelEncoder for this task
enc = preprocessing.LabelEncoder()
df['labels'] = enc.fit_transform(df['names'])
df['labels'] = np.stack([df['labels'][i] + 1 for i in range(len(df['labels']))])

df.head(3)

classes = df[['names', 'labels']].value_counts()
classes

# make dictionary for class objects so we can call objects by their keys.
classes = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

# bounding box coordinates point need to be in separate columns
df['xmin'] = -1
df['ymin'] = -1
df['xmax'] = -1
df['ymax'] = -1

df[['xmin', 'ymin', 'xmax', 'ymax']] = np.stack([df['boxes'][i] for i in range(len(df['boxes']))])

df.drop(columns=['boxes'], inplace=True)
df['xmin'] = df['xmin'].astype(np.float32)
df['ymin'] = df['ymin'].astype(np.float32)
df['xmax'] = df['xmax'].astype(np.float32)
df['ymax'] = df['ymax'].astype(np.float32)

# drop names column since we dont need it anymore
df.drop(columns=['names'], inplace=True)
df.head()

len(df['img_id'].unique())

"""separate train and validation data"""

image_ids = df['img_id'].unique()
test_ids = image_ids[-2000:]
valid_ids = image_ids[-4000:-2000]
train_ids = image_ids[:-4000]
len(train_ids)

train_df = df[df['img_id'].isin(train_ids)]
valid_df = df[df['img_id'].isin(valid_ids)]
test_df = df[df['img_id'].isin(test_ids)]
train_df.shape, valid_df.shape, test_df.shape

"""make dataset by dataset module"""

class VOCDataset(Dataset):
    # Implementation of the dataset class goes here
    pass

def get_transform_train():
    # Implementation of training transformations goes here
    pass

def get_transform_valid():
    # Implementation of validation transformations goes here
    pass

def collate_fn(batch):
    # Implementation of collate function goes here
    pass

train_dataset = VOCDataset(train_df, IMG_PATH, get_transform_train())
valid_dataset = VOCDataset(valid_df, IMG_PATH, get_transform_valid())
test_dataset = VOCDataset(test_df, IMG_PATH, get_transform_valid())

# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()

#train loader
train_data_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)
#valid loader
valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)
#test loader
test_data_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""view samples"""

images, targets = next(iter(train_data_loader))
images = list(image.to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

plt.figure(figsize=(20, 20))
for i, (image, target) in enumerate(zip(images, targets)):
    # Implementation of visualization logic goes here
    pass

"""# Model and Training

download pretrained model
"""

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 21  # 20 classes in the dataset and background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

from engine import train_one_epoch, evaluate

# Custom metrics computation function
def compute_metrics(predictions, annotations, iou_thresholds=None, device='cuda'):
    # Implementation of metrics computation goes here
    pass

# Custom evaluate function
def evaluate(model, data_loader, device):
    # Implementation of evaluation logic goes here
    pass

# Define checkpoint path
checkpoint_path = "model_checkpoint.pth"

# Function to save checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    # Implementation of checkpoint saving logic goes here
    pass

# Function to load checkpoint
def load_checkpoint(model, optimizer, scheduler, path, device):
    # Implementation of checkpoint loading logic goes here
    pass

num_epochs = 8

# Load checkpoint (if it exists) and get the starting epoch
start_epoch = load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path, device)

# Training loop
for epoch in range(start_epoch, num_epochs):
    # Implementation of training loop goes here
    pass

"""Saving the model"""

torch.save(model.state_dict(), 'faster_rcnn_state.pth')
# Save the entire model
torch.save(model, 'best_faster-rcnn_model.pth')

"""test the model"""