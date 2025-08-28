import os
import sys
from PIL import Image
from config import *
from face_detect import face_detect
from age_pred import predict_age
from face_match import check_similarity

device=device
image_dir = input_dir
save_dir = output_dir
images = []
img_count=0
sim_threshold=similarity_threshold

print("Device:", device, "\n")

for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        img_count+=1
        if(img_count>2):
            print("\nYou uploaded more than two images!")
            print("I will work on first two images only :(")
            break
            # sys.exit(0)
        file_path = os.path.join(image_dir, filename)
        images.append(Image.open(file_path).convert("RGB"))    
        print(f"Loaded {filename}")    


faces = face_detect(images, save_dir=save_dir)

ages = predict_age(faces)

print(f"\nFirst  person is: {ages[0]:.2f} years old")
print(f"Second person is: {ages[1]:.2f} years old")

similarity = check_similarity(faces)

print(f"\nSimilarity: {(100*similarity):.2f}%")

if similarity >= sim_threshold:
    print("Faces MATCH .. This is the same person")
else:
    print("Faces do NOT MATCH .. They are two different persons")




























# import numpy as np
# if np.array_equal(np.asarray(faces[0], dtype=np.uint8), np.asarray(Image.open("1_face_detected/face1.png").convert("RGB"), dtype=np.uint8)):
#     print("yeeeeeeeeeees")
# else: 
#     print("Noooo")



# imgs = []
# imgs.append(Image.open("1_face_detected/face1.png").convert('RGB'))
# imgs.append(Image.open("1_face_detected/face2.png").convert('RGB'))
# similarity = check_similarity(imgs)
# print(f"\nSimilarity: {(100*similarity):.2f}%")
# if similarity >= 0.55:
#     print("Faces match .. This is the same person")
# else:
#     print("Faces do not match .. They are two different persons")


# t1 = Image.open("1_face_detected/1.jpg").convert("RGB")
# t2 = Image.open("1_face_detected/2.jpg").convert("RGB")
# t3 = Image.open("1_face_detected/face1.jpg").convert("RGB")
# t4 = Image.open("1_face_detected/face2.jpg").convert("RGB")
# t12 = [t1, t2]
# t34 = [t3, t4]
# ages = predict_age(t12)
# print(ages[0])
# print(ages[1])
# ages = predict_age(t34)
# print(ages[0])
# print(ages[1])