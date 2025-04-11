import os
import random
import time
import cv2
import torch
import yaml
import matplotlib.pyplot as plt
import albumentations as A
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# ---------------------- 1. ตั้งค่าอุปกรณ์ ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------- 2. ตรวจสอบและแสดงตัวอย่างภาพ ----------------------
image_folder = os.path.join("DATASET", "train", "images")
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]


if len(image_files) >= 9:
    plt.figure(figsize=(12, 12))
    transform = A.Compose([
        # 1. ปรับสีสัน และความสว่าง (ทำให้โมเดลเข้าใจภาพในสภาวะแสงต่างๆ)
        A.RandomBrightnessContrast(p=0.3),
        # # A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        # # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),

        # # # 2. การหมุนภาพและพลิกภาพ (ช่วยเรื่องมุมมองที่แตกต่างกัน)
        # # A.Rotate(limit=30, p=0.5),
        # # # A.HorizontalFlip(p=0.5),
        # # # A.VerticalFlip(p=0.2),

        # # # 3. การซูมเข้า-ออก และ Crop (เพิ่มความหลากหลายของเฟรม)
        A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0), ratio=(1.0, 1.0), p=0.60),

        # # 4. การเพิ่ม noise หรือเบลอ (ช่วยให้โมเดลแข็งแกร่งขึ้น)
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        # # A.MotionBlur(blur_limit=5, p=0.2),
        # # A.GaussianBlur(blur_limit=3, p=0.2),

        # # # 5. การเปลี่ยนขนาดภาพแบบ Preserve Aspect Ratio (ช่วยให้โมเดลเรียนรู้ได้ดีขึ้น)
        # # A.Resize(640, 640)
    ])

    for i, image_file in enumerate(random.sample(image_files, 9)):
        img_path = os.path.join(image_folder, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented = transform(image=img_rgb)["image"]
            
            plt.subplot(6, 3, i * 2 + 1)
            plt.imshow(img_rgb)
            plt.title(f"Original: {image_file}")
            plt.axis('off')
            
            plt.subplot(6, 3, i * 2 + 2)
            plt.imshow(augmented)
            plt.title(f"Augmented: {image_file}")
            plt.axis('off')

    plt.show()
else:
    print("Not enough images in the folder!")

# # ---------------------- 3. โหลดโมเดล YOLO พร้อมเพิ่ม Dropout ----------------------
# # เราจะสร้างคลาส CustomYOLO สืบทอดจาก YOLO
## ส่วนนี้จะไว้ค่อยปรับแต่งภายในโมเดล 
# class CustomYOLO(YOLO):
#     def __init__(self, model_path="yolov8n.pt"):
#         super().__init__(model_path)
#         self.batch_norm = nn.BatchNorm2d(256)  # เพิ่ม BatchNorm2D

#     def forward(self, x, augment=False, profile=False):
#         features = self.model.backbone(x)
#         features = self.batch_norm(features)  # ใช้ Batch Normalization
#         features = self.model.neck(features)
#         outputs = self.model.head(features)
#         return outputs

# # # โหลดโมเดลแบบ custom แล้วส่งไปยัง device
# โหลดโมเดล YOLOv9
model = YOLO("yolo12s.pt").to(device)  
# ---------------------- 4. ตรวจสอบและแก้ไข Label Files ----------------------
label_folder = os.path.join("DATASET", "train", "labels")

for label_file in os.listdir(label_folder):
    label_path = os.path.join(label_folder, label_file)
    
    with open(label_path, 'r') as file:
        lines = file.readlines()
    
    cleaned_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            try:
                class_id = int(parts[0])
                if class_id <= 12:
                    cleaned_lines.append(" ".join(parts))
            except ValueError:
                print(f"Invalid class_id in file {label_file}: {line.strip()}")
    
    if cleaned_lines:
        with open(label_path, 'w') as file:
            unique_lines = list(set(cleaned_lines))
            file.write("\n".join(unique_lines) + "\n")
    else:
        os.remove(label_path)  # ลบไฟล์ label ที่ไม่มีข้อมูล

print("✅ ลบคลาสที่เกิน 12 ออกไปหมดแล้ว และลบไฟล์ที่ไม่มีข้อมูล")

# ---------------------- 5. โหลดค่า data.yaml ----------------------
data_yaml = os.path.join("DATASET", "data.yaml")
if not os.path.isfile(data_yaml):
    raise FileNotFoundError(f"Error: {data_yaml} is not a valid file.")
with open(data_yaml, 'r', encoding='utf-8') as file:
    data_config = yaml.safe_load(file)

# ---------------------- 6. สร้าง DataLoader ----------------------
class CustomDataset(Dataset):
    def __init__(self, image_folder, label_folder):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        label_path = os.path.join(self.label_folder, self.image_files[idx].rsplit('.', 1)[0] + '.txt')
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img).float()
        
        labels = torch.zeros((1, 5))  # Default empty label
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = torch.tensor([list(map(float, line.strip().split())) for line in f.readlines()])
        
        return img, labels

dataset = CustomDataset(image_folder=image_folder, label_folder=label_folder)

# ปรับค่า num_workers ให้เป็น 0 เพื่อหลีกเลี่ยงปัญหาใน Windows
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

# ---------------------- 7. ฝึกโมเดล YOLO ----------------------
if __name__ == "__main__":
    epochs = 100
    start_time = time.time()
    
    # เรียกใช้การฝึกโมเดล
    model.train(
        data=data_yaml,         epochs=epochs,  
        batch=32,  
        lr0=0.001,
        lrf=0.1,
        workers=0,  # ตั้งค่า workers เป็น 0 สำหรับ Windows
        device=device,
        imgsz=640,  
        conf=0.30,  
        iou=0.45
    )
    
    # ประเมินผลโมเดล
    results = model.val(
        data=data_yaml,
        device=device,
        conf=0.30,  
        iou=0.45
    )
    
    print(f"Training & Validation ใช้เวลา: {time.time() - start_time:.2f} วินาที")
