import torch, numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from config import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

device = device
model_path = face_matching_model_path

# pretrained VGGFace2 for 512d embeddings
model = InceptionResnetV1(num_classes=8631).eval().to(device)
state_dict = torch.load(model_path)
if 'logits.weight' in state_dict:
    state_dict.pop('logits.weight')
if 'logits.bias' in state_dict:
    state_dict.pop('logits.bias')
model.load_state_dict(state_dict, strict=False)
# model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# preprocess for FaceNet (160x160 and map [0,1] -> [-1,1]
preprocess = transforms.Compose([
    transforms.Resize((160, 160)),                   
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1,1] (prewhiten)
])

@torch.inference_mode()
def embed_face(img):
    # img = Image.open(path).convert('RGB')
    x = preprocess(img).unsqueeze(0).to(device) # [1,3,160,160]
    emb = model(x) # [1,512]
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # L2-normalize
    return emb.squeeze(0).cpu().numpy()

def cos_similarity(a, b):
    return float(np.dot(a, b))  

def check_similarity(imgs):
    e1, e2 = embed_face(imgs[0]), embed_face(imgs[1])
    cos_sim = cos_similarity(e1, e2)
    return cos_sim

