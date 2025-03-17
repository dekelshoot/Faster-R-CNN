# filepath: /home/dekelshoot/Bureau/faster_rcnn/src/test_model.py
"""test the model on a set of images"""

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as T
import matplotlib.pyplot as plt
import sys

# make dictionary for class objects so we can call objects by their keys.
classes = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load a model; pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

WEIGHTS_FILE = "faster_rcnn_state.pth"
num_classes = 21

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the trained weights
model.load_state_dict(torch.load(WEIGHTS_FILE))
model = model.to(device)
model.eval()

# Helper function to visualize bounding boxes
def visualize_predictions(image, boxes, labels, scores, threshold=0.5):
    # Filter boxes with scores above the threshold
    keep_indices = scores > threshold
    boxes = boxes[keep_indices]
    labels = labels[keep_indices]
    scores = scores[keep_indices]

    # Map class numbers to names
    label_names = [classes.get(label.item(), f"Class {label.item()}") for label in labels]

    # Draw bounding boxes
    image = T.ToPILImage()(image).convert("RGB")
    image_tensor = T.ToTensor()(image)
    drawn_image = draw_bounding_boxes(
        image_tensor,
        boxes,
        labels=[f"{name} ({score:.2f})" for name, score in zip(label_names, scores)],
        colors="green",
        width=2,
    )
    return drawn_image

# Example usage in the visualization loop
image_paths = sys.argv[1:]  # Get image paths from command line arguments

for image_path in image_paths:
    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        outputs = model(image_tensor)

    # Unpack predictions for the first image in the batch
    predicted_boxes = outputs[0]["boxes"].cpu()
    predicted_labels = outputs[0]["labels"].cpu()
    predicted_scores = outputs[0]["scores"].cpu()

    # Visualize predictions
    drawn_image = visualize_predictions(
        image_tensor[0].cpu(),
        predicted_boxes,
        predicted_labels,
        predicted_scores,
        threshold=0.5
    )

    # Display the image with bounding boxes
    plt.figure(figsize=(12, 8))
    plt.imshow(drawn_image.permute(1, 2, 0))
    plt.axis("off")
    plt.title(f"Predictions for {image_path}")
    plt.show()