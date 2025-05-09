import os
import glob
from skimage.transform import warp, ProjectiveTransform
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
path = "1/hpatches-sequences-release/"
validation_data_base = []
for dirs in os.listdir(path):
    current_set = []
    # I'm interested only in prespective changes so I chose only view point sets not intensity
    if dirs.startswith("i"): continue
    img_path1 = path + dirs + "/1.ppm"
    for i in range(2, 7):
        img_path2 = path + dirs + f"/{i}.ppm"
        transformation = path + dirs + f"/H_{1}_{i}"
        current_set.append((img_path1, img_path2, transformation))
    validation_data_base.append(current_set)
    
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)
def apply_homography(image, H):
    h, w = image.shape[:2]
    return cv2.warpPerspective(image, H, (w, h))

def extract_keypoints_and_matches(image0, image1, extractor, matcher):
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    return m_kpts0.cpu().numpy(), m_kpts1.cpu().numpy(), matches01

def extract_patches(image, keypoints, patch_size=32):
    patches = []
    h, w = image.shape[:2]
    half = patch_size // 2
    for (x, y) in keypoints:
        x, y = int(x), int(y)
        x1, y1 = max(0, x - half), max(0, y - half)
        x2, y2 = min(w, x + half), min(h, y + half)
        patch = image[y1:y2, x1:x2]
        patch = cv2.resize(patch, (patch_size, patch_size))
        patch = patch.astype(np.float32) / 255.0
        if patch.ndim == 3 and patch.shape[2] == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patches.append(patch)
    return np.array(patches)
patch_size = 50
num_train_images = 50
torch.set_grad_enabled(True)
import numpy as np
import torch
from torch.utils.data import Dataset

# class SinglePatchDataset(Dataset):
#     def __init__(self, npz_path):
#         data = np.load(npz_path)
#         self.patches0 = data["patches0"]  
#         self.offsets = data["offsets"]   
#     def __len__(self):
#         return len(self.offsets)

#     def __getitem__(self, idx):

#         patch0 = self.patches0[idx]
#         offset = self.offsets[idx] 


#         patch_tensor = torch.from_numpy(patch0).unsqueeze(0).float() 
#         offset_tensor = torch.from_numpy(offset).float()              

#         return patch_tensor, offset_tensor
# class DualPatchDataset(Dataset):
#     def __init__(self, npz_path):
#         data = np.load(npz_path)
#         self.patches0 = data["patches0"]  
#         self.patches1 = data["patches1"]  
#         self.offsets = data["offsets"]    
        
#     def __len__(self):
#         return len(self.offsets)

#     def __getitem__(self, idx):
#         patch0 = self.patches0[idx]  
#         patch1 = self.patches1[idx]

#         offset = self.offsets[idx]   

#         patch_pair = np.stack([patch0, patch1, patch1-patch0], axis=0)

#         patch_tensor = torch.from_numpy(patch_pair).float() 
#         offset_tensor = torch.from_numpy(offset).float()

#         return patch_tensor, offset_tensor
def crop_patch_centered(image, center, size):
    """
    image: numpy array, shape (H, W)
    center: tuple or array-like (x, y) keypoint 좌표 (픽셀 단위)
    size: int, 추출할 정사각형 패치의 크기 (예: 8, 20)
    """
    H, W = image.shape
    half_size = size // 2
    cx, cy = int(round(center[0])), int(round(center[1]))
    
    # 시작 인덱스 계산 (좌상단)
    start_x = cx - half_size
    start_y = cy - half_size
    end_x = start_x + size
    end_y = start_y + size

    # 만약 crop 영역이 이미지 범위를 벗어나면 패딩 처리 (여기서는 0으로 패딩)
    pad_left = max(0, -start_x)
    pad_top = max(0, -start_y)
    pad_right = max(0, end_x - W)
    pad_bottom = max(0, end_y - H)
    
    if pad_left or pad_top or pad_right or pad_bottom:
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
        start_x += pad_left
        start_y += pad_top
        end_x += pad_left
        end_y += pad_top

    return image[start_y:end_y, start_x:end_x]

class DualPatchDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.patches0 = data["patches0"]  # 원본 이미지, shape (H, W) 또는 (H, W, 채널)일 수 있음
        self.offsets = data["offsets"]    # keypoint 좌표, 예: [x, y]

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        patch0 = self.patches0[idx]
        offset = self.offsets[idx]  # keypoint 좌표 (x, y)

        # 만약 patch0가 (H, W, 채널)인 경우, 그레이스케일로 변환하거나 채널 0만 사용할 수 있음
        # 여기서는 patch0가 단일 채널 이미지라고 가정
        if patch0.ndim == 3:
            # 예를 들어, 첫번째 채널을 사용 (또는 필요에 따라 변환)
            patch0 = patch0[..., 0]

        # keypoint를 중심으로 8x8 및 20x20 패치 추출
        patch_small = crop_patch_centered(patch0, offset, 8)
        patch_large = crop_patch_centered(patch0, offset, 20)

        # 3채널로 확장
        patch_tensor_small = torch.from_numpy(np.repeat(patch_small[..., np.newaxis], 3, axis=-1)).permute(2, 0, 1).float()
        patch_tensor_large = torch.from_numpy(np.repeat(patch_large[..., np.newaxis], 3, axis=-1)).permute(2, 0, 1).float()

        offset_tensor = torch.from_numpy(offset).float()

        return patch_tensor_small, patch_tensor_large, offset_tensor


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class KeypointRefiner(nn.Module):
    def __init__(self, in_channels=3, patch_size=30):  
        super().__init__()

        self.layer1 = ResNetBasicBlock(in_channels, 32)  
        self.layer2 = ResNetBasicBlock(32, 64, stride=2) 
        self.layer3 = ResNetBasicBlock(64, 128, stride=2)  
        
        self.fc1 = None  
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = x.view(x.size(0), -1)  

        if self.fc1 is None:
            input_dim = x.shape[1] 
            self.fc1 = nn.Linear(input_dim, 256).to(x.device)  

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        out = self.fc3(x)

        return out

from torch.utils.data import DataLoader
import torch.optim as optim

train_ds = DualPatchDataset("homography_train_dataset_50.npz")
val_ds   = DualPatchDataset("homography_val_dataset_50.npz")

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetKeypointRefiner(in_channels=6).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.SmoothL1Loss()

epochs = 15
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for patch_small, patch_large, offsets in train_loader:
        patch_small = patch_small.to(device)
        patch_large = patch_large.to(device)
        offsets = offsets.to(device)
        
        optimizer.zero_grad()
        pred_coords, heatmap = model(patch_small, patch_large)
        loss = criterion(pred_coords, offsets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    train_loss = running_loss / len(train_loader)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for patch_small, patch_large, offsets in val_loader:
            patch_small = patch_small.to(device)
            patch_large = patch_large.to(device)
            offsets = offsets.to(device)
            
            pred_coords, heatmap = model(patch_small, patch_large)
            loss = criterion(pred_coords, offsets)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    
    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
import numpy as np
import torch
import cv2
import torch.nn.functional as F

def refine_keypoints_with_model(
    img1, img2,
    m_kpts0, m_kpts1,
    model,
    patch_size=16,
    device='cuda'
):
    model.eval()
    corrected_kpts1 = []

    for kp0, kp1 in zip(m_kpts0, m_kpts1):
        patch0 = extract_single_patch(img1, kp0, patch_size)
        patch1 = extract_single_patch(img2, kp1, patch_size)

        if patch0.ndim == 2:
            patch0 = np.expand_dims(patch0, axis=0)
        else:
            patch0 = np.transpose(patch0, (2, 0, 1))
            
        if patch1.ndim == 2:
            patch1 = np.expand_dims(patch1, axis=0)
        else:
            patch1 = np.transpose(patch1, (2, 0, 1))
        
        patch0_tensor = torch.from_numpy(patch0).unsqueeze(0).float().to(device)
        patch1_tensor = torch.from_numpy(patch1).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            pred_offset = model(patch0_tensor, patch1_tensor)

        offset_np = pred_offset[0].cpu().numpy()
        
        kp1_corrected = kp1 + offset_np
        corrected_kpts1.append(kp1_corrected)

    return np.array(corrected_kpts1)


def extract_single_patch(image, keypoint, patch_size=16):
    x, y = int(keypoint[0]), int(keypoint[1])
    half = patch_size // 2
    h, w = image.shape[:2]

    x1, x2 = max(0, x - half), min(w, x + half)
    y1, y2 = max(0, y - half), min(h, y + half)

    patch = image[y1:y2, x1:x2]
    patch = cv2.resize(patch, (patch_size, patch_size))
    patch = patch.astype(np.float32) / 255.0

    if patch.ndim == 2:
        patch = np.stack([patch, patch, patch], axis=-1)
    elif patch.ndim == 3 and patch.shape[2] != 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    
    return patch



def compute_refinement_error(gt_kpts, refined_kpts):
    distances = np.linalg.norm(gt_kpts - refined_kpts, axis=1)
    mean_error = np.mean(distances)
    std_error = np.std(distances)
    return mean_error, std_error


def process_validation_data(validation_data_base, extractor, matcher, model, patch_size=16, device='cuda'):
    all_mean_errors_before = []
    all_std_errors_before = []
    all_mean_errors_after = []
    all_std_errors_after = []
    
    for image_set in validation_data_base:
        for img1_path, img2_path, tr_path in image_set:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)   
            transformation = np.loadtxt(tr_path)
            
            image1_torch = torch.from_numpy(img1.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            image2_torch = torch.from_numpy(img2.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            
            feats0 = extractor.extract(image1_torch)
            feats1 = extractor.extract(image2_torch)
            matches01 = matcher({"image0": feats0, "image1": feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
            
            kpts0 = feats0["keypoints"].cpu().numpy()
            kpts1 = feats1["keypoints"].cpu().numpy()
            matches = matches01["matches"]
            if isinstance(matches, torch.Tensor):
                matches = matches.cpu().numpy()
            
            m_kpts0 = kpts0[matches[:, 0]]
            m_kpts1 = kpts1[matches[:, 1]]
            
            ones = np.ones((m_kpts0.shape[0], 1), dtype=np.float32)
            kpts0_homogeneous = np.hstack([m_kpts0, ones])  
            kpts1_gt = (transformation @ kpts0_homogeneous.T).T  
            kpts1_gt[:, :2] /= kpts1_gt[:, 2:].copy()
            kpts1_gt = kpts1_gt[:, :2]  
            
            refined_kpts1 = refine_keypoints_with_model(
                img1, img2, m_kpts0, m_kpts1,
                model=model,
                patch_size=patch_size,
                device=device
            )
            
            mean_err_before, std_err_before = compute_refinement_error(kpts1_gt, m_kpts1)
            mean_err_after, std_err_after = compute_refinement_error(kpts1_gt, refined_kpts1)
            
            all_mean_errors_before.append(mean_err_before)
            all_std_errors_before.append(std_err_before)
            all_mean_errors_after.append(mean_err_after)
            all_std_errors_after.append(std_err_after)
            
    return (
        np.mean(all_mean_errors_before), np.std(all_std_errors_before),
        np.mean(all_mean_errors_after), np.std(all_std_errors_after)
    )

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mean_before, std_before, mean_after, std_after = process_validation_data(
    validation_data_base, extractor, matcher, model,
    patch_size=50,
    device=device
)
print(f"Before refinement - Mean error: {mean_before:.2f} pixels, Std error: {std_before:.2f} pixels")
print(f"After refinement - Mean error: {mean_after:.2f} pixels, Std error: {std_after:.2f} pixels")
