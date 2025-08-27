import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

age_pred_model_path = "models/resnet50_age.pth"

input_dir = '0_input_images'
output_dir = '1_face_detected'       # or None

# cosine similarity threshold
threshold = 0.50