import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import requests
import shutil
from tqdm import tqdm
import subprocess

# ===================== Siamese Dataset cho CEDAR =====================
class SiameseCEDARDataset(Dataset):
    def __init__(self, cedar_dir, transform=None):
        # Lưu transform (augmentation, resize, normalize, ...)
        self.transform = transform
        # Tạo dict lưu các ảnh theo từng class (writer_id)
        self.samples_by_class = {}

        # Đường dẫn tới thư mục chứa chữ ký thật
        org_dir = os.path.join(cedar_dir, 'signatures', 'full_org')
        print(f"Đang đọc dữ liệu từ thư mục: {org_dir}")
        
        if not os.path.exists(org_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục {org_dir}")

        # Liệt kê tất cả các file trong thư mục
        all_files = os.listdir(org_dir)
        print(f"Tổng số file trong thư mục: {len(all_files)}")
        print("Danh sách 5 file đầu tiên:")
        for f in all_files[:5]:
            print(f"- {f}")

        # Đọc tất cả các file ảnh trong thư mục
        for img_name in all_files:
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                # Lấy writer_id từ tên file (giả sử format là: original_X_Y.png)
                try:
                    parts = img_name.split('_')
                    if len(parts) >= 2:
                        writer_id = int(parts[1])
                        img_path = os.path.join(org_dir, img_name)
                        if writer_id not in self.samples_by_class:
                            self.samples_by_class[writer_id] = []
                        self.samples_by_class[writer_id].append(img_path)
                except (ValueError, IndexError) as e:
                    print(f"Bỏ qua file {img_name} do không đúng định dạng: {str(e)}")
                    continue

        if not self.samples_by_class:
            raise ValueError(f"Không tìm thấy file ảnh nào trong thư mục {org_dir}")

        print(f"Đã tìm thấy {len(self.samples_by_class)} người ký khác nhau")
        for writer_id, samples in self.samples_by_class.items():
            print(f"Người ký {writer_id}: {len(samples)} mẫu")

        # Danh sách các class (writer_id)
        self.classes = list(self.samples_by_class.keys())
        # Sinh các cặp ảnh (positive/negative)
        self.pairs = self.generate_pairs()
        print(f"Tổng số cặp ảnh được tạo: {len(self.pairs)}")

    def generate_pairs(self):
        pairs = []
        for class_id in self.classes:
            samples = self.samples_by_class[class_id]
            # Sinh các cặp positive (cùng người)
            for i in range(len(samples) - 1):
                pairs.append((samples[i], samples[i + 1], 1))
            # Sinh các cặp negative (khác người)
            other_classes = [c for c in self.classes if c != class_id]
            for i in range(len(samples)):
                neg_class = random.choice(other_classes)
                neg_sample = random.choice(self.samples_by_class[neg_class])
                pairs.append((samples[i], neg_sample, 0))
        return pairs

    def __len__(self):
        # Tổng số cặp ảnh
        return len(self.pairs)

    def __getitem__(self, idx):
        # Lấy đường dẫn 2 ảnh và label (1: cùng người, 0: khác người)
        img1_path, img2_path, label = self.pairs[idx]
        # Đọc ảnh RGB
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        # Áp dụng transform nếu có
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # Trả về 2 tensor ảnh và label dạng float32
        return img1, img2, torch.tensor(label, dtype=torch.float32)

# ===================== Kiến trúc SigNet (Siamese branch) =====================
class SigNetBase(nn.Module):
    def __init__(self):
        super(SigNetBase, self).__init__()
        # Đầu vào 3 kênh (RGB)
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
        self.fc1 = nn.Linear(256 * 3 * 5, 1024)

        self.fc2 = nn.Linear(1024, 128)  # Embedding 128 chiều

    def forward(self, x):
        # Block conv1 + BN + ReLU + Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Block conv2 + BN + ReLU + Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Block conv3 + BN + ReLU
        x = F.relu(self.bn3(self.conv3(x)))
        # Block conv4 + BN + ReLU
        x = F.relu(self.bn4(self.conv4(x)))
        # Block conv5 + BN + ReLU + Pool
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        # Flatten
        x = x.view(x.size(0), -1)
        # FC1 + ReLU
        x = F.relu(self.fc1(x))
        # FC2 (embedding)
        x = self.fc2(x)
        return x

# ===================== Contrastive Loss =====================
def contrastive_loss(output1, output2, label, margin=1.0):
    # Tính khoảng cách Euclidean giữa 2 embedding
    distance = F.pairwise_distance(output1, output2)
    # Loss cho cặp giống nhau: label*distance^2, cho cặp khác: (1-label)*relu(margin-distance)^2
    loss = label * distance.pow(2) + (1 - label) * F.relu(margin - distance).pow(2)
    return loss.mean()

# ===================== Hàm tính accuracy cho Siamese =====================
def compute_accuracy(output1, output2, label, threshold=0.5):
    # Tính khoảng cách Euclidean giữa 2 embedding
    distance = F.pairwise_distance(output1, output2)
    # Nếu distance < threshold thì dự đoán là cùng người (1), ngược lại là khác người (0)
    pred = (distance < threshold).float()
    # So sánh với ground truth
    correct = (pred == label).float().sum()
    # Tính tỷ lệ đúng
    acc = correct / label.size(0)
    return acc.item()

# ===================== Hàm train Siamese SigNet =====================
def train_siamese_model(data_dir, num_epochs=20, batch_size=16, lr=1e-4):
    # Transform cho ảnh RGB: resize, augment, normalize
    transform = transforms.Compose([
        transforms.Resize((155, 220)),  # Kích thước chuẩn của SigNet gốc
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Augmentation
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # Chuẩn hóa 3 kênh RGB
    ])

    # Tạo dataset Siamese
    dataset = SiameseCEDARDataset(data_dir, transform=transform)
    # Chia train/val
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx))

    # Khởi tạo model và optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SigNetBase().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    threshold = 0.5  # Ngưỡng để phân biệt positive/negative

    # Vòng lặp huấn luyện
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            out1 = model(img1)
            out2 = model(img2)
            loss = contrastive_loss(out1, out2, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # Tính accuracy cho batch này
            train_acc += compute_accuracy(out1, out2, label, threshold)
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # Đánh giá trên validation
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                out1 = model(img1)
                out2 = model(img2)
                loss = contrastive_loss(out1, out2, label)
                val_loss += loss.item()
                # Tính accuracy cho batch này
                val_acc += compute_accuracy(out1, out2, label, threshold)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

    # Lưu model sau khi train xong
    torch.save(model.state_dict(), "signature_model.pt")
    print("\n✅ Training complete. Model saved to signature_model.pt")

def download_file(url, filename):
    """
    Tải file với thanh tiến trình
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def extract_rar(rar_file, output_dir):
    """
    Giải nén file RAR sử dụng UnRAR
    """
    # Đường dẫn mặc định của UnRAR trên Windows
    unrar_paths = [
        r"C:\Program Files\WinRAR\UnRAR.exe",
        r"C:\Program Files (x86)\WinRAR\UnRAR.exe",
        "unrar"  # Thử tìm trong PATH
    ]
    
    for unrar_path in unrar_paths:
        try:
            subprocess.run([unrar_path, 'x', rar_file, output_dir], check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    
    print("Không tìm thấy UnRAR. Vui lòng cài đặt WinRAR từ:")
    print("https://www.rarlab.com/rar_add.htm")
    print("Sau khi cài đặt, hãy khởi động lại terminal và chạy lại script.")
    return False

def download_and_process_cedar(data_dir):
    """
    Tải và xử lý dataset CEDAR
    """
    print(f"Bắt đầu tải và xử lý dataset CEDAR vào thư mục: {data_dir}")
    
    # URL của dataset CEDAR
    cedar_url = "http://www.cedar.buffalo.edu/NIJ/data/signatures.rar"
    
    # Tạo cấu trúc thư mục
    signatures_dir = os.path.join(data_dir, "signatures")
    full_org_dir = os.path.join(signatures_dir, "full_org")
    full_forg_dir = os.path.join(signatures_dir, "full_forg")
    
    # Xóa thư mục cũ nếu tồn tại
    if os.path.exists(signatures_dir):
        print(f"Xóa thư mục cũ: {signatures_dir}")
        shutil.rmtree(signatures_dir)
    
    # Tạo thư mục mới
    os.makedirs(full_org_dir, exist_ok=True)
    os.makedirs(full_forg_dir, exist_ok=True)
    
    # Tải dataset
    print("Đang tải dataset CEDAR...")
    download_file(cedar_url, "signatures.rar")
    
    # Giải nén file
    print("Đang giải nén dữ liệu...")
    temp_dir = "temp_cedar"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    if not extract_rar("signatures.rar", temp_dir):
        return False
    
    # Kiểm tra nội dung thư mục tạm
    print("\nNội dung thư mục tạm sau khi giải nén:")
    for root, dirs, files in os.walk(temp_dir):
        print(f"\nThư mục: {root}")
        print(f"Số file: {len(files)}")
        if files:
            print("5 file đầu tiên:")
            for f in files[:5]:
                print(f"- {f}")
    
    # Tổ chức lại cấu trúc thư mục
    print("\nĐang tổ chức lại cấu trúc thư mục...")
    moved_files = 0
    
    # Tìm và di chuyển các file
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, file)
                # Kiểm tra cả tên file và tên thư mục
                if any(keyword in root.lower() for keyword in ['genuine', 'original', 'real']):
                    dst_path = os.path.join(full_org_dir, file)
                elif any(keyword in root.lower() for keyword in ['forg', 'fake']):
                    dst_path = os.path.join(full_forg_dir, file)
                else:
                    # Nếu không xác định được, dựa vào tên file
                    if any(keyword in file.lower() for keyword in ['genuine', 'original', 'real']):
                        dst_path = os.path.join(full_org_dir, file)
                    else:
                        dst_path = os.path.join(full_forg_dir, file)
                
                try:
                    shutil.copy2(src_path, dst_path)  # Sử dụng copy2 thay vì move
                    moved_files += 1
                except Exception as e:
                    print(f"Lỗi khi di chuyển file {file}: {str(e)}")
    
    print(f"\nĐã di chuyển {moved_files} file")
    print(f"Số file trong full_org: {len(os.listdir(full_org_dir))}")
    print(f"Số file trong full_forg: {len(os.listdir(full_forg_dir))}")
    
    # Kiểm tra kết quả
    if not os.listdir(full_org_dir):
        print("Lỗi: Không có file nào trong thư mục full_org")
        return False
    if not os.listdir(full_forg_dir):
        print("Lỗi: Không có file nào trong thư mục full_forg")
        return False
    
    # Tạo file Readme.txt
    readme_content = """CEDAR Signature Dataset
Original signatures in full_org
Forged signatures in full_forg
"""
    with open(os.path.join(signatures_dir, "Readme.txt"), "w") as f:
        f.write(readme_content)
    
    # Dọn dẹp
    shutil.rmtree(temp_dir)
    os.remove("signatures.rar")
    
    print("Hoàn thành việc tải và xử lý dataset CEDAR!")
    return True

def check_and_organize_data(data_dir):
    """
    Kiểm tra và tổ chức lại dữ liệu nếu cần
    """
    print(f"Kiểm tra dữ liệu trong thư mục: {data_dir}")
    
    # Tạo cấu trúc thư mục nếu chưa có
    signatures_dir = os.path.join(data_dir, "signatures")
    full_org_dir = os.path.join(signatures_dir, "full_org")
    full_forg_dir = os.path.join(signatures_dir, "full_forg")
    
    os.makedirs(full_org_dir, exist_ok=True)
    os.makedirs(full_forg_dir, exist_ok=True)
    
    # Kiểm tra xem có file nào trong thư mục gốc không
    root_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if root_files:
        print(f"Tìm thấy {len(root_files)} file trong thư mục gốc, đang di chuyển...")
        for file in root_files:
            src_path = os.path.join(data_dir, file)
            if 'genuine' in file.lower() or 'original' in file.lower():
                dst_path = os.path.join(full_org_dir, file)
            else:
                dst_path = os.path.join(full_forg_dir, file)
            shutil.move(src_path, dst_path)
    
    # Kiểm tra các thư mục con
    for root, dirs, files in os.walk(data_dir):
        if root == data_dir or root == signatures_dir:
            continue
            
        if files:
            print(f"Tìm thấy {len(files)} file trong {root}")
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    src_path = os.path.join(root, file)
                    if 'genuine' in root.lower() or 'original' in root.lower():
                        dst_path = os.path.join(full_org_dir, file)
                    else:
                        dst_path = os.path.join(full_forg_dir, file)
                    shutil.move(src_path, dst_path)
    
    # Kiểm tra kết quả
    org_files = os.listdir(full_org_dir)
    forg_files = os.listdir(full_forg_dir)
    
    print(f"\nKết quả tổ chức dữ liệu:")
    print(f"- Số file trong full_org: {len(org_files)}")
    print(f"- Số file trong full_forg: {len(forg_files)}")
    
    if not org_files and not forg_files:
        raise ValueError("Không tìm thấy file ảnh nào trong dataset!")
    
    return True

# ===================== Hàm main để chạy qua dòng lệnh =====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Siamese SigNet with Contrastive Loss (RGB)")
    parser.add_argument("--data_dir", type=str, default="data/cedar", help="Thư mục chứa dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Số epoch train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--download", action="store_true", help="Tải dataset CEDAR nếu chưa có")
    args = parser.parse_args()

    # Tải dataset nếu cần
    if args.download:
        if not download_and_process_cedar(args.data_dir):
            print("Không thể tải dataset CEDAR. Vui lòng kiểm tra lại kết nối mạng và thử lại.")
            exit(1)
    else:
        # Kiểm tra và tổ chức lại dữ liệu nếu cần
        if not check_and_organize_data(args.data_dir):
            print("Không thể tổ chức lại dữ liệu. Vui lòng kiểm tra lại thư mục dataset.")
            exit(1)

    # Gọi hàm train
    train_siamese_model(args.data_dir, args.epochs, args.batch_size, args.lr)

def verify_signature(model_path, input_image_path, reference_dir, class_names):
    # ... existing code ...
    
    try:
        input_signature = detect_signature(input_image_path, save_detection=True)
        # Thêm debug info
        print(f"Kích thước ảnh sau detect: {input_signature.size}")
        
        input_tensor = transform(input_signature)
        print(f"Kích thước tensor: {input_tensor.shape}")
        print(f"Kênh màu: {input_tensor.shape[0]}")
        
        input_embedding = get_embedding(model, input_tensor)
        print(f"Kích thước embedding: {input_embedding.shape}")
        
    except Exception as e:
        print(f"Lỗi chi tiết: {str(e)}")
        return None, 0.0
