# 📤 Hướng dẫn đẩy lên GitHub & Google Colab

---

## Phần A: Đẩy lên GitHub

### Bước 1: Tạo Repository trên GitHub

1. Truy cập **[github.com](https://github.com)** → Đăng nhập
2. Nhấn nút **"+"** (góc trên phải) → **"New repository"**
3. Điền thông tin:
   - **Repository name:** `FashionMNIST`
   - **Description:** `Deep Learning Image Classification - FashionMNIST`
   - **Visibility:** `Public` hoặc `Private`
   - ⚠️ **KHÔNG tick** "Add a README file" (vì ta đã có sẵn)
4. Nhấn **"Create repository"**
5. **Copy URL** của repo (dạng `https://github.com/<USERNAME>/FashionMNIST.git`)

> [!IMPORTANT]
> Ghi nhớ URL repo, sẽ dùng ở Bước 3.

---

### Bước 2: Cài đặt Git (nếu chưa có)

Mở **Terminal / PowerShell** và kiểm tra:

```bash
git --version
```

Nếu chưa có, tải tại: [https://git-scm.com/downloads](https://git-scm.com/downloads)

Cấu hình Git lần đầu:

```bash
git config --global user.name "Tên của bạn"
git config --global user.email "email@example.com"
```

---

### Bước 3: Khởi tạo Git & Push lên GitHub

Mở **Terminal / PowerShell**, chạy lần lượt từng lệnh:

```bash
# 1. Di chuyển vào thư mục project
cd D:\Code\Deep_Learning\FashionMNIST

# 2. Khởi tạo Git repository
git init

# 3. Thêm tất cả file vào staging
git add .

# 4. Kiểm tra các file sẽ được commit (nên kiểm tra)
git status

# 5. Tạo commit đầu tiên
git commit -m "feat: initial commit - FashionMNIST project"

# 6. Đổi tên branch thành main
git branch -M main

# 7. Kết nối với GitHub (thay <USERNAME> bằng tên GitHub của bạn)
git remote add origin https://github.com/<USERNAME>/FashionMNIST.git

# 8. Push lên GitHub
git push -u origin main
```

> [!NOTE]
> Lần đầu push, GitHub sẽ yêu cầu đăng nhập. Nhập **username** và **Personal Access Token** (không phải password).

---

### Bước 4: Tạo Personal Access Token (nếu cần)

Nếu GitHub yêu cầu xác thực:

1. Vào **GitHub** → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
2. Nhấn **"Generate new token (classic)"**
3. Đặt tên: `FashionMNIST-access`
4. Tick quyền: `repo` (full control of private repositories)
5. Nhấn **"Generate token"**
6. **Copy token** → Dùng thay password khi push

---

### Bước 5: Cập nhật code sau này

Mỗi khi có thay đổi, chạy:

```bash
cd D:\Code\Deep_Learning\FashionMNIST

# Xem file nào thay đổi
git status

# Thêm file thay đổi
git add .

# Commit với mô tả
git commit -m "feat: thêm model CNN"

# Push lên GitHub
git push
```

#### Quy ước Commit Message:

| Prefix | Dùng khi |
|--------|----------|
| `feat:` | Thêm tính năng mới |
| `fix:` | Sửa lỗi |
| `docs:` | Cập nhật tài liệu |
| `test:` | Thêm/sửa test |
| `refactor:` | Refactor code |

**Ví dụ:**
```bash
git commit -m "feat: thêm data augmentation pipeline"
git commit -m "fix: sửa lỗi normalize transform"
git commit -m "docs: cập nhật README"
```

---

### Bước 6: Mời thành viên nhóm (Collaborators)

1. Vào **repo trên GitHub** → **Settings** → **Collaborators**
2. Nhấn **"Add people"**
3. Nhập **username GitHub** của từng thành viên: Bảo, Bình, Khoa, Việt, Hưng
4. Họ sẽ nhận email mời → Nhấn **Accept**

Sau khi được mời, các thành viên clone repo:

```bash
git clone https://github.com/<USERNAME>/FashionMNIST.git
cd FashionMNIST
```

---

## Phần B: Sử dụng trên Google Colab

### Cách 1: Mở Notebook trực tiếp từ GitHub (Nhanh nhất ⚡)

1. Truy cập: **[colab.research.google.com](https://colab.research.google.com)**
2. Chọn tab **"GitHub"**
3. Dán URL repo: `https://github.com/<USERNAME>/FashionMNIST`
4. Chọn notebook **`FashionMNIST_Colab.ipynb`** → Nhấn mở

Hoặc truy cập trực tiếp bằng URL:
```
https://colab.research.google.com/github/<USERNAME>/FashionMNIST/blob/main/FashionMNIST_Colab.ipynb
```

---

### Cách 2: Upload Notebook lên Google Drive

1. Mở **Google Drive** → Tạo thư mục `FashionMNIST`
2. Upload file `FashionMNIST_Colab.ipynb` vào thư mục đó
3. Double-click file `.ipynb` → Tự động mở bằng Google Colab

---

### Cách 3: Clone repo trong Colab

Tạo notebook mới trên Colab và chạy:

```python
# Clone repo từ GitHub
!git clone https://github.com/<USERNAME>/FashionMNIST.git
%cd FashionMNIST

# Xem cấu trúc thư mục
!ls -la
```

---

### Thiết lập GPU trên Colab

> [!IMPORTANT]
> Bắt buộc bật GPU để train nhanh hơn!

1. Vào menu **Runtime** → **Change runtime type**
2. Chọn **Hardware accelerator:** `T4 GPU`
3. Nhấn **Save**
4. Kiểm tra:
```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # Tesla T4
```

---

### Lưu kết quả từ Colab về Google Drive

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy model đã train vào Drive
!cp best_model.pth "/content/drive/MyDrive/FashionMNIST/"
!cp fashion_mnist_full_checkpoint.pth "/content/drive/MyDrive/FashionMNIST/"

print("✅ Đã lưu model vào Google Drive!")
```

---

### Lưu thay đổi từ Colab ngược lên GitHub

```python
# Cấu hình Git trong Colab
!git config --global user.name "Tên của bạn"
!git config --global user.email "email@example.com"

# Thêm và commit
!git add .
!git commit -m "feat: thêm kết quả training từ Colab"

# Push (cần token)
!git push https://<TOKEN>@github.com/<USERNAME>/FashionMNIST.git main
```

> [!CAUTION]
> Không hard-code token vào notebook khi repo là Public. Sử dụng Colab Secrets:
> 1. Nhấn icon 🔑 (bên trái Colab) → Add new secret
> 2. Name: `GITHUB_TOKEN`, Value: `ghp_xxxxxxxxxxxx`
> 3. Trong code:
> ```python
> from google.colab import userdata
> token = userdata.get('GITHUB_TOKEN')
> !git push https://{token}@github.com/<USERNAME>/FashionMNIST.git main
> ```

---

## Phần C: Chia sẻ cho nhóm cùng làm trên Colab

### Chia sẻ Notebook

1. Mở notebook trên Colab
2. Nhấn **"Share"** (góc trên phải)
3. Thêm email các thành viên: Bảo, Bình, Khoa, Việt, Hưng
4. Chọn quyền **"Editor"** (Người chỉnh sửa)
5. Nhấn **"Send"**

### Quy tắc làm việc trên Colab

| Quy tắc | Mô tả |
|---------|-------|
| **Mỗi người 1 bản copy** | `File > Save a copy in Drive` để tránh conflict |
| **Đặt tên rõ ràng** | `FashionMNIST_Bao_EDA.ipynb`, `FashionMNIST_Binh_Training.ipynb` |
| **Commit thường xuyên** | Push kết quả lên GitHub sau mỗi phiên làm việc |
| **Ghi chú rõ ràng** | Thêm Markdown cell mô tả mỗi đoạn code |

---

## 📁 Tóm tắt các file trong project

| File | Mô tả |
|------|-------|
| `README.md` | Mô tả dự án & workflow |
| `TEAM_ASSIGNMENT.md` | Phân công công việc nhóm |
| `GITHUB_COLAB_GUIDE.md` | Hướng dẫn này |
| `FashionMNIST_Colab.ipynb` | Notebook chạy trên Colab |
| `.gitignore` | Loại bỏ file không cần push |
| `raw/` | Dữ liệu FashionMNIST gốc (không push) |

---

## ❓ Xử lý lỗi thường gặp

### Lỗi: `fatal: remote origin already exists`
```bash
git remote remove origin
git remote add origin https://github.com/<USERNAME>/FashionMNIST.git
```

### Lỗi: `Authentication failed`
→ Kiểm tra lại Personal Access Token (đã hết hạn hay chưa)

### Lỗi: `failed to push some refs`
```bash
git pull origin main --rebase
git push origin main
```

### Colab bị ngắt kết nối
→ Colab tự ngắt sau ~90 phút idle. Lưu checkpoint thường xuyên!

```python
# Thêm vào training loop
if epoch % 5 == 0:
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pth')
```

