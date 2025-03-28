from flask import Flask, render_template, request, send_from_directory
import os
import librosa
import numpy as np
from db_connection import get_db_connection
import webbrowser
from werkzeug.utils import secure_filename
import tempfile
import time
import threading

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# Cấu hình
ALLOWED_EXTENSIONS = {'wav'}
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../static/uploads"))
AUDIO_FILE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../audio/audio0"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Giới hạn 10MB

# Lưu trữ thông tin file tạm và thời gian lưu trữ
temp_files = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_temp_files():
    """Dọn dẹp các file tạm đã hết hạn"""
    while True:
        current_time = time.time()
        expired_files = []
        
        for filepath, expiry_time in list(temp_files.items()):
            if current_time > expiry_time:
                if os.path.exists(filepath):
                    try:
                        os.unlink(filepath)
                        print(f"Đã xóa file tạm: {filepath}")
                    except:
                        pass
                expired_files.append(filepath)
        
        for filepath in expired_files:
            temp_files.pop(filepath, None)
            
        time.sleep(30)  # Kiểm tra mỗi 30 giây

def extract_features(file_path):
    """Trích xuất đặc trưng của file âm thanh"""
    try:
        y, sr = librosa.load(file_path, duration=30)  # Giới hạn 30s để xử lý nhanh
        
        # Đặc trưng tần số cơ bản
        f0 = librosa.pitch_tuning(y) + librosa.estimate_tuning(y=y, sr=sr)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        
        # Năng lượng
        energy = np.sum(y**2)
        
        # Ghép thành vector đặc trưng
        return np.concatenate([[f0], mfcc_mean, [centroid_mean, zcr_mean, energy]])
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {str(e)}")
        return None

def search_similar_files(input_features):
    """Tìm kiếm các file tương đồng nhất từ cơ sở dữ liệu"""
    conn, status = get_db_connection()
    if conn is None:
        return None, status
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT filename, instrument_name, 
                fundamental_freq, 
                mfcc_1, mfcc_2, mfcc_3, mfcc_4, mfcc_5,
                mfcc_6, mfcc_7, mfcc_8, mfcc_9, mfcc_10, 
                mfcc_11, mfcc_12, mfcc_13, 
                spectral_centroid, zcr, energy
            FROM wind_instruments
        """)
        
        results = []
        for row in cursor.fetchall():
            # Tạo vector đặc trưng từ dữ liệu trong DB
            db_features = np.array([
                row['fundamental_freq'],
                row['mfcc_1'], row['mfcc_2'], row['mfcc_3'], 
                row['mfcc_4'], row['mfcc_5'], row['mfcc_6'], 
                row['mfcc_7'], row['mfcc_8'], row['mfcc_9'],
                row['mfcc_10'], row['mfcc_11'], row['mfcc_12'],
                row['mfcc_13'],
                row['spectral_centroid'],
                row['zcr'],
                row['energy']
            ])
            
            # Tính toán khoảng cách Euclidean
            distance = np.linalg.norm(input_features - db_features)
            
            results.append({
                'filename': row['filename'],
                'instrument_name': row['instrument_name'],
                'distance': distance,
                'audio_url': f"/audio/{row['filename']}"  # Đường dẫn để nghe file
            })
        
        # Sắp xếp theo khoảng cách tăng dần (gần nhất trước)
        return sorted(results, key=lambda x: x['distance'])[:3], "Thành công"
    except Exception as e:
        return None, f"Lỗi tìm kiếm: {str(e)}"
    finally:
        if conn and conn.is_connected():
            conn.close()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error_message="Không có file nào được tải lên")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error_message="Chưa chọn file")
        
        if not allowed_file(file.filename):
            return render_template('index.html', error_message="Chỉ chấp nhận file WAV")
        
        # Lưu file tạm thời
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        try:
            # Trích xuất đặc trưng từ file tải lên
            features = extract_features(temp_path)
            if features is None:
                return render_template('index.html', error_message="Lỗi xử lý file âm thanh")
            
            # Trích xuất tên nhạc cụ từ tên file
            instrument_name = ''.join(filter(str.isalpha, filename.split('.')[0]))
            
            # Tìm kiếm các file tương đồng
            results, status = search_similar_files(features)
            if results is None:
                return render_template('index.html', error_message=status)
                
            # Đánh dấu file tạm cần giữ lại với thời gian hết hạn sau 5 phút
            temp_files[temp_path] = time.time() + 60  # 1 phút
            
            return render_template('index.html', 
                                 input_filename=filename,
                                 input_instrument_name=instrument_name,
                                 input_file_url=f"/uploads/{filename}",
                                 results=results)
                                 
        except Exception as e:
            # Xóa file nếu xảy ra lỗi
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return render_template('index.html', error_message=f"Lỗi: {str(e)}")
    
    return render_template('index.html')

@app.route("/audio/<path:filename>")
def serve_audio(filename):
    """Phục vụ các file âm thanh từ thư mục train"""
    full_path = os.path.join(AUDIO_FILE_DIR, filename)
    print(f"Đang phục vụ file âm thanh: {full_path}, tồn tại: {os.path.exists(full_path)}")
    if not os.path.exists(full_path):
        print(f"Thư mục hiện tại: {os.getcwd()}")
        # print(f"Nội dung thư mục {AUDIO_FILE_DIR}: {os.listdir(AUDIO_FILE_DIR) if os.path.exists(AUDIO_FILE_DIR) else 'không tồn tại'}")
    return send_from_directory(AUDIO_FILE_DIR, filename)

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    """Phục vụ các file âm thanh tải lên"""
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"Đang phục vụ file tạm: {full_path}, tồn tại: {os.path.exists(full_path)}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    # Khởi động luồng dọn dẹp file tạm
    cleanup_thread = threading.Thread(target=cleanup_temp_files, daemon=True)
    cleanup_thread.start()
    
    print("Khởi động ứng dụng tìm kiếm nhạc cụ...")
    url = "http://127.0.0.1:5000"
    webbrowser.open(url)
    app.run(host="127.0.0.1", port=5000, debug=False)