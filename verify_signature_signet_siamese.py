import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import torch.nn as nn

# ===== SigNet Siamese Embedding Model =====
class SigNetBase(nn.Module):
    def __init__(self):
        super(SigNetBase, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.bn1 = nn.BatchNorm2d(96)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(384)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 3 * 5, 1024)  # Phù hợp với ảnh 155x220 sau conv
        self.fc2 = nn.Linear(1024, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ===== Trích xuất embedding từ ảnh =====
def get_embedding(model, image_tensor):
    model.eval()
    with torch.no_grad():
        x = image_tensor.unsqueeze(0)
        x = model(x)
        x = F.normalize(x, p=2, dim=1)
        return x.cpu().numpy().squeeze()

# ===== Phát hiện và cắt vùng chữ ký (CẢI TIẾN) =====
def detect_signature(image_path, save_detection=True):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]
    if not valid_contours:
        raise ValueError("Không tìm thấy chữ ký hợp lệ trong ảnh")

    # Gộp tất cả các contour hợp lệ thành 1 vùng bao duy nhất
    all_points = np.vstack(valid_contours)
    x, y, w, h = cv2.boundingRect(all_points)

    # (Không kiểm tra aspect_ratio nữa)

    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2*padding)
    h = min(image.shape[0] - y, h + 2*padding)
    signature = image[y:y+h, x:x+w]

    if save_detection:
        os.makedirs("imgDetection", exist_ok=True)
        base_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        box_path = os.path.join("imgDetection", f"{name_without_ext}_box.jpg")
        cut_path = os.path.join("imgDetection", f"{name_without_ext}_detected.jpg")
        image_boxed = image.copy()
        cv2.rectangle(image_boxed, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite(box_path, image_boxed)
        cv2.imwrite(cut_path, signature)

    signature_rgb = cv2.cvtColor(signature, cv2.COLOR_BGR2RGB)
    return Image.fromarray(signature_rgb)

# ===== Load model đã huấn luyện =====
def load_model(model_path):
    model = SigNetBase()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# ===== Xác minh chữ ký với các thư mục tham chiếu =====
def verify_signature(model_path, input_image_path, reference_dir, class_names):
    SIMILARITY_THRESHOLD = 0.90
    GENUINE_THRESHOLD = 0.90
    MIN_REFERENCE_SIMILARITY = 0.85

    try:
        input_signature = detect_signature(input_image_path, save_detection=True)
    except Exception as e:
        print(f"Lỗi phát hiện chữ ký: {e}")
        return None, 0.0

    transform = transforms.Compose([
        transforms.Resize((155, 220)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    model = load_model(model_path)
    input_tensor = transform(input_signature)
    input_embedding = get_embedding(model, input_tensor)

    best_similarity = -1
    best_match_class = None
    reference_consistencies = {}

    for class_name in class_names:
        person_dir = os.path.join(reference_dir, class_name)
        embeddings = []

        for file in os.listdir(person_dir):
            if file.endswith((".png", ".jpg")):
                try:
                    ref_img = detect_signature(os.path.join(person_dir, file), save_detection=False)
                    ref_tensor = transform(ref_img)
                    ref_embedding = get_embedding(model, ref_tensor)
                    embeddings.append(ref_embedding)
                    print(f"[OK] {file}")
                except:
                    print(f"[ERROR] Lỗi khi xử lý {file}: {e}")
                    continue

        if len(embeddings) < 2:
            continue

        similarities = [cosine_similarity([e1], [e2])[0][0] for i, e1 in enumerate(embeddings) for e2 in embeddings[i+1:]]
        mean_ref_sim = np.mean(similarities)
        print(f"[DEBUG] Reference consistency for {class_name}: {mean_ref_sim:.4f}")

        reference_consistencies[class_name] = mean_ref_sim
        if mean_ref_sim < MIN_REFERENCE_SIMILARITY:
            print(f"[WARNING] Low consistency in {class_name} (mean sim = {mean_ref_sim:.4f})")


        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding /= np.linalg.norm(mean_embedding)
        similarity = cosine_similarity([input_embedding], [mean_embedding])[0][0]

        if similarity > best_similarity:
            best_similarity = similarity
            best_match_class = class_name

    if best_similarity < SIMILARITY_THRESHOLD:
        print("Predicted: Unknown Person")
        print(f"Best Similarity: {best_similarity:.4f}")
        print("Genuine: NO")
    else:
        print(f"Predicted: {best_match_class}")
        print(f"Similarity: {best_similarity:.4f}")
        print(f"Reference Consistency: {reference_consistencies.get(best_match_class, 0):.4f}")
        print(f"Genuine: {'YES' if best_similarity >= GENUINE_THRESHOLD else 'NO'}")

    return best_match_class, best_similarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify signature using Siamese SigNet")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--reference_dir", type=str, required=True)
    parser.add_argument("--class_names", nargs='+', required=True)
    args = parser.parse_args()

    verify_signature(args.model_path, args.input_image, args.reference_dir, args.class_names) 