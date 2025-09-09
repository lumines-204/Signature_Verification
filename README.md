# Signature Verification System

Hệ thống xác thực chữ ký sử dụng mô hình Siamese Network với dataset CEDAR.

## Cấu trúc thư mục

```
Signature_Verification/
│
├── data/
│   └── cedar/
│       └── signatures/
│           ├── full_org/     # Chứa chữ ký thật từ dataset CEDAR
│           ├── full_forg/    # Chứa chữ ký giả từ dataset CEDAR
│           └── Readme.txt    # File thông tin dataset
│
├── reference_signatures/     # Thư mục chứa chữ ký tham chiếu
│   ├── person1_id/          # Thư mục chứa chữ ký của người 1
│   │   ├── signature1.jpg
│   │   ├── signature2.jpg
│   │   └── ...
│   └── person2_id/          # Thư mục chứa chữ ký của người 2
│       ├── signature1.jpg
│       ├── signature2.jpg
│       └── ...
│
├── Signature_Input/ # Thư mục chứa các ảnh chữ ký cần áp dụng xác thực
│   ├── input_signature1.jpg
│   ├── input_signature2.jpg
│   └── ...
│
├── imgDetection/            # Thư mục lưu kết quả phát hiện chữ ký
│   ├── *_box.jpg           # Ảnh gốc với khung đánh dấu vùng chữ ký
│   └── *_detected.jpg      # Ảnh chữ ký đã được cắt ra
│
├── train_signature_model.py           # Script huấn luyện mô hình
├── verify_signature_signet_siamese.py # Script xác minh chữ ký
├── signature_model.pt                 # File model đã huấn luyện
```

## Yêu cầu hệ thống

1. Python 3.7 trở lên
2. PyTorch
3. WinRAR hoặc 7-Zip (để giải nén dataset)
4. Các thư viện Python cần thiết:
   - torch
   - torchvision
   - PIL
   - numpy
   - scikit-learn
   - tqdm
   - requests

## Cài đặt

1. Cài đặt Python và các thư viện cần thiết:
```bash
    pip install torch torchvision pillow numpy scikit-learn tqdm requests
```

2. Cài đặt WinRAR hoặc 7-Zip để giải nén dataset

## Cách sử dụng

### 1. Tải và chuẩn bị dataset

```bash
    python train_signature_model.py --download --data_dir data/cedar
```

Lệnh này sẽ:
- Tải dataset CEDAR
- Giải nén và tổ chức lại cấu trúc thư mục
- Tạo các thư mục full_org và full_forg

### 2. Huấn luyện mô hình

```bash
    python train_signature_model.py --data_dir data/cedar --epochs 10 --batch_size 4 --lr 0.0001
```

Các tham số:
- `--data_dir`: Thư mục chứa dataset (mặc định: data/cedar)
- `--epochs`: Số epoch huấn luyện (mặc định: 20)
- `--batch_size`: Kích thước batch (mặc định: 16)
- `--lr`: Learning rate (mặc định: 0.0001)

### 3. Xác minh chữ ký

```bash
    python verify_signature_signet_siamese.py --model_path "signature_model.pt" --input_image "Signature_Input/input_signature.jpg" --reference_dir "reference_signatures" --class_names "person1_id" "person2_id"
```

Các tham số:
- `--model_path`: Đường dẫn đến file model đã huấn luyện
- `--input_image`: Đường dẫn đến ảnh chữ ký cần xác minh
- `--reference_dir`: Thư mục chứa chữ ký tham chiếu
- `--class_names`: Danh sách ID của những người cần so sánh

### 4. Kết quả

1. Sau khi huấn luyện:
   - Model sẽ được lưu vào file `signature_model.pt`
   - Kết quả huấn luyện sẽ hiển thị trên terminal

2. Sau khi xác minh chữ ký:
   - Kết quả phát hiện sẽ được lưu trong thư mục `imgDetection`
   - Kết quả xác minh sẽ hiển thị trên terminal:
     + Người sở hữu chữ ký
     + Độ tương đồng (similarity score)
     + Kết luận về tính xác thực (genuine/forged)

## Lưu ý

1. Đảm bảo có đủ dung lượng ổ đĩa (khoảng 1GB) để tải và giải nén dataset
2. Quá trình huấn luyện có thể mất nhiều thời gian tùy thuộc vào phần cứng
3. Nếu gặp lỗi khi giải nén, hãy kiểm tra xem đã cài đặt WinRAR hoặc 7-Zip chưa
4. Ảnh đầu vào nên có độ tương phản tốt giữa chữ ký và nền
5. Có thể điều chỉnh các ngưỡng trong code để thay đổi độ nhạy của hệ thống