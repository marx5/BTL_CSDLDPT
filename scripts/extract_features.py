
import os
import librosa
import numpy as np
import pandas as pd
import shutil
import random

# Sử dụng đường dẫn tuyệt đối dựa trên vị trí của file script hiện tại
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # Lên một cấp từ thư mục scripts
AUDIO_SOURCE_DIR = os.path.join(PROJECT_DIR, "audio/audio0")  # Thư mục nguồn chứa tất cả file âm thanh
AUDIO_TRAIN_DIR = os.path.join(PROJECT_DIR, "audio/audio_train")  # Thư mục lưu file train
AUDIO_TEST_DIR = os.path.join(PROJECT_DIR, "audio/audio_test")  # Thư mục lưu file test
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train_features.csv")

# Tỷ lệ chia train/test (0.8 = 80% train, 20% test)
TRAIN_RATIO = 0.8

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    f0 = librosa.pitch_tuning(y) + librosa.estimate_tuning(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    energy = np.sum(y**2)
    
    filename = os.path.basename(file_path)
    instrument_name = ''.join(filter(str.isalpha, filename.split('.')[0]))
    
    features = {
        "filename": filename,
        "instrument_name": instrument_name,
        "fundamental_freq": f0,
        **{f"mfcc_{i+1}": mfcc_mean[i] for i in range(13)},
        "spectral_centroid": centroid_mean,
        "zcr": zcr_mean,
        "energy": energy
    }
    return features

def split_audio_files():
    """Chia các file âm thanh thành hai tập train và test"""
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(AUDIO_TRAIN_DIR, exist_ok=True)
    os.makedirs(AUDIO_TEST_DIR, exist_ok=True)
    
    # Xóa các file cũ trong thư mục train và test (để tránh trùng lặp)
    for file in os.listdir(AUDIO_TRAIN_DIR):
        os.remove(os.path.join(AUDIO_TRAIN_DIR, file))
    for file in os.listdir(AUDIO_TEST_DIR):
        os.remove(os.path.join(AUDIO_TEST_DIR, file))
    
    # Lấy danh sách các file âm thanh
    audio_files = [f for f in os.listdir(AUDIO_SOURCE_DIR) if f.endswith('.wav')]
    
    if not audio_files:
        print(f"Không tìm thấy file âm thanh nào trong {AUDIO_SOURCE_DIR}")
        return
    
    # Xáo trộn file để đảm bảo tính ngẫu nhiên
    random.shuffle(audio_files)
    
    # Tính số lượng file cho tập train
    train_size = int(len(audio_files) * TRAIN_RATIO)
    
    # Chia file vào các thư mục tương ứng
    train_files = audio_files[:train_size]
    test_files = audio_files[train_size:]
    
    print(f"Đang chia tệp: {len(train_files)} files cho train, {len(test_files)} files cho test")
    
    # Copy file vào thư mục train
    for file in train_files:
        src = os.path.join(AUDIO_SOURCE_DIR, file)
        dst = os.path.join(AUDIO_TRAIN_DIR, file)
        shutil.copy2(src, dst)
        
    # Copy file vào thư mục test
    for file in test_files:
        src = os.path.join(AUDIO_SOURCE_DIR, file)
        dst = os.path.join(AUDIO_TEST_DIR, file)
        shutil.copy2(src, dst)
    
    print(f"Đã hoàn tất việc chia tập dữ liệu!")
    return train_files, test_files

def extract_features_for_train():
    """Trích xuất đặc trưng chỉ cho các file trong tập train"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Xóa file CSV cũ nếu tồn tại
    if os.path.exists(TRAIN_CSV):
        os.remove(TRAIN_CSV)
        print(f"Đã xóa {TRAIN_CSV}")
    
    features_list = []
    for filename in os.listdir(AUDIO_TRAIN_DIR):
        if filename.endswith(".wav"):
            file_path = os.path.join(AUDIO_TRAIN_DIR, filename)
            print(f"Đang xử lý file train: {filename}")
            features = extract_features(file_path)
            features_list.append(features)
    
    if not features_list:
        print("Không tìm thấy file .wav nào trong thư mục train!")
    else:
        df = pd.DataFrame(features_list)
        
        # Lưu đặc trưng vào train.csv
        df.to_csv(TRAIN_CSV, index=False)
        print(f"Đã lưu {len(df)} file vào bộ train {TRAIN_CSV}")

if __name__ == "__main__":
    print("Bắt đầu tiến trình chia tập dữ liệu và trích xuất đặc trưng...")
    # Bước 1: Chia file âm thanh thành train và test
    train_files, test_files = split_audio_files()
    
    # Bước 2: Chỉ trích xuất đặc trưng cho tập train
    extract_features_for_train()
    
    print("Hoàn tất quá trình!")


    # import os
# import librosa
# import numpy as np
# import pandas as pd

# # Sử dụng đường dẫn tuyệt đối dựa trên vị trí của file script hiện tại
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # Lên một cấp từ thư mục scripts
# AUDIO_FILE_DIR = os.path.join(PROJECT_DIR, "audio2")
# DATA_DIR = os.path.join(PROJECT_DIR, "data")
# TRAIN_CSV = os.path.join(DATA_DIR, "train_features.csv")

# def extract_features(file_path):
#     y, sr = librosa.load(file_path)
#     f0 = librosa.pitch_tuning(y) + librosa.estimate_tuning(y=y, sr=sr)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     mfcc_mean = np.mean(mfcc, axis=1)
#     centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
#     centroid_mean = np.mean(centroid)
#     zcr = librosa.feature.zero_crossing_rate(y)
#     zcr_mean = np.mean(zcr)
#     energy = np.sum(y**2)
    
#     filename = os.path.basename(file_path)
#     instrument_name = ''.join(filter(str.isalpha, filename.split('.')[0]))
    
#     features = {
#         "filename": filename,
#         "instrument_name": instrument_name,
#         "fundamental_freq": f0,
#         **{f"mfcc_{i+1}": mfcc_mean[i] for i in range(13)},
#         "spectral_centroid": centroid_mean,
#         "zcr": zcr_mean,
#         "energy": energy
#     }
#     return features

# if __name__ == "__main__":
#     os.makedirs(DATA_DIR, exist_ok=True)
    
#     # Xóa file CSV cũ nếu tồn tại
#     if os.path.exists(TRAIN_CSV):
#         os.remove(TRAIN_CSV)
#         print(f"Đã xóa {TRAIN_CSV}")
    
#     features_list = []
#     for filename in os.listdir(AUDIO_FILE_DIR):
#         if filename.endswith(".wav"):
#             file_path = os.path.join(AUDIO_FILE_DIR, filename)
#             print(f"Đang xử lý file train: {filename}")
#             features = extract_features(file_path)
#             features_list.append(features)
    
#     if not features_list:
#         print("Không tìm thấy file .wav nào trong thư mục audio!")
#     else:
#         df = pd.DataFrame(features_list)
        
#         # Lưu toàn bộ dữ liệu vào train.csv, không chia tập test
#         df.to_csv(TRAIN_CSV, index=False)
#         print(f"Đã lưu {len(df)} file vào bộ train {TRAIN_CSV}")
