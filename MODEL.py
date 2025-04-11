import os
import random
import time
import cv2
import torch
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import albumentations as A
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset

# ---------------------- 1. ตั้งค่าอุปกรณ์ ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------- 2. ตรวจสอบและแสดงตัวอย่างภาพ ----------------------
image_folder = os.path.join("DATASET", "train", "images")
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

if len(image_files) >= 9:
    plt.figure(figsize=(12, 12))
    transform = A.Compose([
         A.RandomBrightnessContrast(p=0.2),
        # A.VerticalFlip(p=0.2),
         A.Rotate(limit=(90, 180), p=0.5),  # หมุนภาพระหว่าง 90 ถึง 100 องศา
         A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0), ratio=(1.0, 1.0), p=0.40)
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

# ---------------------- 3. โหลดโมเดล YOLO ----------------------
model = YOLO("yolov8n.pt").to(device)

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
    epochs = 10
    start_time = time.time()
    
    # เรียกใช้การฝึกโมเดล
    model.train(
        data=data_yaml, 
        epochs=epochs,  
        batch=32,  
        lr0=0.001,  
        workers=0,  # ตั้งค่า workers เป็น 0 สำหรับ Windows
        device=device,
        imgsz=640,  
        conf=0.20,  
        iou=0.45
    )
    
    # ประเมินผลโมเดล
    results = model.val(
        data=data_yaml,
        device=device,
        conf=0.20,  
        iou=0.45
    )
    
    print(f"Training & Validation ใช้เวลา: {time.time() - start_time:.2f} วินาที")

def display_images(post_training_files_path, image_files):
    for image_file in image_files:
        image_path = os.path.join(post_training_files_path, image_file)

        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image: {image_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 10), dpi=120)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

# List of image files to display
image_files = [
    'confusion_matrix_normalized.png',
    'F1_curve.png',
    'P_curve.png',
    'R_curve.png',
    'PR_curve.png',
    'results.png'
]

# Path to the directory containing the images
post_training_files_path = r"D:\\MODEL_AI\\DATASET\\train"

# Display the images
display_images(post_training_files_path, image_files)

Result_Final_model = pd.read_csv("D:\\Users\\user\\Desktop\\MODEL_AI\\runs\\detect\\train\\results.csv")
Result_Final_model.tail(10)

# Read the results.csv file as a pandas dataframe
Result_Final_model.columns = Result_Final_model.columns.str.strip()

# Create subplots
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))

# Plot the columns using seaborn
sns.lineplot(x='epoch', y='train/box_loss', data=Result_Final_model, ax=axs[0,0])
sns.lineplot(x='epoch', y='train/cls_loss', data=Result_Final_model, ax=axs[0,1])
sns.lineplot(x='epoch', y='train/dfl_loss', data=Result_Final_model, ax=axs[1,0])
sns.lineplot(x='epoch', y='metrics/precision(B)', data=Result_Final_model, ax=axs[1,1])
sns.lineplot(x='epoch', y='metrics/recall(B)', data=Result_Final_model, ax=axs[2,0])
sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=Result_Final_model, ax=axs[2,1])
sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=Result_Final_model, ax=axs[3,0])
sns.lineplot(x='epoch', y='val/box_loss', data=Result_Final_model, ax=axs[3,1])
sns.lineplot(x='epoch', y='val/cls_loss', data=Result_Final_model, ax=axs[4,0])
sns.lineplot(x='epoch', y='val/dfl_loss', data=Result_Final_model, ax=axs[4,1])

# Set titles and axis labels for each subplot
axs[0,0].set(title='Train Box Loss')
axs[0,1].set(title='Train Class Loss')
axs[1,0].set(title='Train DFL Loss')
axs[1,1].set(title='Metrics Precision (B)')
axs[2,0].set(title='Metrics Recall (B)')
axs[2,1].set(title='Metrics mAP50 (B)')
axs[3,0].set(title='Metrics mAP50-95 (B)')
axs[3,1].set(title='Validation Box Loss')
axs[4,0].set(title='Validation Class Loss')
axs[4,1].set(title='Validation DFL Loss')


plt.suptitle('Training Metrics and Loss', fontsize=24)
plt.subplots_adjust(top=0.8)
plt.tight_layout()
plt.show()



# ฟังก์ชันสำหรับการบันทึกรูปภาพพร้อมผลลัพธ์
def save_image_with_prediction(image, prediction, save_dir, image_name, result_data):
    # เพิ่มผลการทำนายบนภาพ
    annotated_image = prediction[0].plot(line_width=1)
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # สร้างโฟลเดอร์เพื่อบันทึกรูปถ้ายังไม่มี
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # บันทึกรูปภาพที่มีผลการทำนาย
    save_path = os.path.join(save_dir, image_name)
    cv2.imwrite(save_path, annotated_image_rgb)
    print(f"Image saved to {save_path}")
    
    # บันทึกผลทำนายในไฟล์ .txt
    txt_path = os.path.join(save_dir, image_name.replace('.jpg', '.txt'))
    with open(txt_path, 'w') as f:
        for item in result_data:
            f.write(f"{item}\n")
    
    # บันทึกผลทำนายในไฟล์ .json
    json_path = os.path.join(save_dir, image_name.replace('.jpg', '.json'))
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=4)
    print(f"Results saved to {txt_path} and {json_path}")


if __name__ == '__main__':
    # โหลดโมเดล
    Valid_model = YOLO('D:\\Users\\user\\Desktop\\MODEL_AI\\runs\\detect\\train\\weights\\best.pt')

    # ประเมินโมเดลบนชุดข้อมูลทดสอบ (เปลี่ยนเป็น test แทน val)
    metrics = Valid_model.val(split='test', workers=0)

    # แสดงผลลัพธ์
    print("precision(B): ", metrics.results_dict["metrics/precision(B)"])
    print("metrics/recall(B): ", metrics.results_dict["metrics/recall(B)"])
    print("metrics/mAP50(B): ", metrics.results_dict["metrics/mAP50(B)"])
    print("metrics/mAP50-95(B): ", metrics.results_dict["metrics/mAP50-95(B)"])

    # ฟังก์ชันสำหรับการปรับขนาดภาพ
    def normalize_image(image):
        return image / 255.0

    def resize_image(image, size=(640, 640)):
        return cv2.resize(image, size)

    # เส้นทางไปยังชุดข้อมูลทดสอบ
    dataset_path = 'D:\\MODEL_AI\\DATASET'
    test_images_path = os.path.join(dataset_path, 'test', 'images')
    save_dir = 'D:\\MODEL_AI\\runs\\detect\\train'

    # ตรวจสอบว่าเส้นทางมีอยู่หรือไม่
    if not os.path.exists(test_images_path):
        print(f"Path not found: {test_images_path}")
    else:
        image_files = [file for file in os.listdir(test_images_path) if file.endswith('.jpg')]

        if len(image_files) > 0:
            print(f"Found {len(image_files)} images in {test_images_path}")

            num_images = len(image_files)
            step_size = max(1, num_images // 16)  # Ensure the interval is at least 1
            selected_images = [image_files[i] for i in range(0, num_images, step_size)]

            fig, axes = plt.subplots(4, 4, figsize=(20, 21))
            fig.suptitle('Testing Set Inferences', fontsize=24)

            for i, ax in enumerate(axes.flatten()):
                if i < len(selected_images):
                    image_path = os.path.join(test_images_path, selected_images[i])
                    image = cv2.imread(image_path)

                    if image is not None:
                        resized_image = resize_image(image, size=(640, 640))
                        normalized_image = normalize_image(resized_image)
                        normalized_image_uint8 = (normalized_image * 255).astype(np.uint8)

                        results = Valid_model.predict(source=normalized_image_uint8, imgsz=640, conf=0.5)
                        annotated_image = results[0].plot(line_width=1)
                        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                        result_data = []
                        for det in results[0].boxes:
                            class_idx = int(det.cls)
                            class_name = Valid_model.names[class_idx]  # ชื่อคลาส
                            confidence = det.conf
                            confidence = confidence.item()

                            xmin, ymin, xmax, ymax = det.xyxy[0].tolist()

                            label = f"{class_name} {confidence:.2f}"
                            cv2.putText(annotated_image_rgb, label, (int(xmin), int(ymin) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                            cv2.rectangle(annotated_image_rgb, (int(xmin), int(ymin)),
                                          (int(xmax), int(ymax)), (0, 255, 0), 2)

                            result_data.append({
                                "class": class_name,
                                "confidence": confidence,
                                "bbox": [xmin, ymin, xmax, ymax]
                            })

                        ax.imshow(annotated_image_rgb)
                        save_image_with_prediction(image, results, save_dir, selected_images[i], result_data)
                    else:
                        print(f"Failed to load image {image_path}")
                ax.axis('off')

            plt.tight_layout()
            plt.show()
        else:
            print(f"No images found in the directory: {test_images_path}")


