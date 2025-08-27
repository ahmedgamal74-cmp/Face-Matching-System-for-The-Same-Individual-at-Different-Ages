from facenet_pytorch import MTCNN
from PIL import Image
import torch
import os, pathlib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from config import *

image_size = 224
device=device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

mtcnn = MTCNN(
    image_size=image_size,
    margin=40,                
    keep_all=False,
    select_largest=True,
    post_process=False,      
    device=device,
)

def face_detect(imgs, save_dir=None):
    face1 = mtcnn(imgs[0])            
    face2 = mtcnn(imgs[1]) 

    if(face1 is None or face2 is None):
        raise RuntimeError("Could not detect faces !")

    if face1.dtype != torch.uint8:
        face1 = face1.clamp(0, 255).byte()
    if face2.dtype != torch.uint8:
        face2 = face2.clamp(0, 255).byte()

    face_img1 = Image.fromarray(face1.permute(1, 2, 0).cpu().numpy(), mode="RGB")
    face_img2 = Image.fromarray(face2.permute(1, 2, 0).cpu().numpy(), mode="RGB")

    faces = []
    faces.append(face_img1)   
    faces.append(face_img2)   

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True) 
        for name in ("face1.png", "face2.png"):
            path = os.path.join(save_dir, name)
            if os.path.isfile(path):
                os.remove(path)
        faces[0].save(f"{save_dir}/face1.png")   
        faces[1].save(f"{save_dir}/face2.png")
        print(f"\nCropped Faces were saved at {save_dir}")

    return faces     

