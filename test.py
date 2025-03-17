# filepath: /home/dekelshoot/Bureau/faster_rcnn/src/test.py

import sys
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# make dictionary for class objects so we can call objects by their keys.
classes = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

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

# Vérifier les arguments de la ligne de commande
if len(sys.argv) != 2:
    print("Usage: python test.py <image_path>")
    sys.exit(1)

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

# Charger une image
image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")

# Transformer l'image
transform = T.Compose([
    T.ToTensor(),
])
image_tensor = transform(image).unsqueeze(0).to(device)  # Ajouter la dimension batch et déplacer vers le device

# Mettre le modèle en mode évaluation
model.eval()

# Effectuer la prédiction
with torch.no_grad():
    outputs = model(image_tensor)

# Déballer les prédictions
predicted_boxes = outputs[0]["boxes"].cpu()
predicted_labels = outputs[0]["labels"].cpu()
predicted_scores = outputs[0]["scores"].cpu()

# Visualiser les prédictions
drawn_image = visualize_predictions(
    image_tensor[0].cpu(), 
    predicted_boxes, 
    predicted_labels, 
    predicted_scores, 
    threshold=0.5
)

# Enregistrer l'image avec les boîtes englobantes
output_path = 'output.jpeg'
drawn_image_pil = T.ToPILImage()(drawn_image)
drawn_image_pil.save(output_path)
print(f"Image saved to {output_path}")