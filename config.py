import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

age_pred_model_path = "models/resnet50_age.pth"
face_matching_model_path = "models/vggface2_recognition.pt"

input_dir = '0_input_images'
output_dir = '1_face_detected'       # put it (None) if you do not need to save detected faces

similarity_threshold = 0.50    # cosine similarity threshold