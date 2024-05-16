import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image

# Load the model architectures
xception = torchvision.models.xception(pretrained=False)
efficientnet_b4 = models.efficientnet_b4(pretrained=False)
efficientnet_autoatt_b4 = models.efficientnet_autoatt_b4(pretrained=False)


# Load the models and weights
xception = torch.load('weights/binclass/net-Xception_traindb-celebdf_face-scale_size-224_seed-41/bestval.pth')
efficientnet_b4 = torch.load('weights/binclass/net-EfficientNetB4_traindb-celebdf_face-scale_size-224_seed-41/bestval.pth')
efficientnet_autoatt_b4 = torch.load('weights/binclass/net-EfficientNetAutoAttB4_traindb-celebdf_face-scale_size-224_seed-41/bestval.pth')

# Load the state_dict into the models
xception.load_state_dict(xception_weights)
efficientnet_b4.load_state_dict(efficientnet_b4_weights)
efficientnet_autoatt_b4.load_state_dict(efficientnet_autoatt_b4_weights)

# Set the models to evaluation mode
xception.eval()
efficientnet_b4.eval()
efficientnet_autoatt_b4.eval()

# Define the transform for the input video frames
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])





# Function to predict the video
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Initialize the predictions
    xception_preds = []
    efficientnet_b4_preds = []
    efficientnet_autoatt_b4_preds = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to PIL Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Apply the transform
        img_tensor = transform(img).unsqueeze(0)
        
        # Make predictions
        with torch.no_grad():
            xception_output = xception(img_tensor)
            efficientnet_b4_output = efficientnet_b4(img_tensor)
            efficientnet_autoatt_b4_output = efficientnet_autoatt_b4(img_tensor)
        
        # Append the predictions
        xception_preds.append(xception_output.item())
        efficientnet_b4_preds.append(efficientnet_b4_output.item())
        efficientnet_autoatt_b4_preds.append(efficientnet_autoatt_b4_output.item())
    
    # Calculate the average predictions
    xception_avg_pred = np.mean(xception_preds)
    efficientnet_b4_avg_pred = np.mean(efficientnet_b4_preds)
    efficientnet_autoatt_b4_avg_pred = np.mean(efficientnet_autoatt_b4_preds)
    
    # Determine the final prediction
    if xception_avg_pred > 0.5 and efficientnet_b4_avg_pred > 0.5 and efficientnet_autoatt_b4_avg_pred > 0.5:
        return "Fake"
    else:
        return "Real"

# Example usage
video_path = 'E:/MastersProject/Celeb-DF-v2/Celeb-synthesis/id0_id1_0000.mp4'
prediction = predict_video(video_path)
print(f"The video is: {prediction}")