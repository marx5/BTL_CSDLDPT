import pandas as pd
import os
from db_connection import get_db_connection

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # Lên một cấp từ thư mục scripts
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train_features.csv")

def create_table():
    conn, status = get_db_connection()
    if conn is None:
        print(status)
        return
    
    cursor = conn.cursor()
    # Xóa bảng cũ nếu tồn tại
    cursor.execute("DROP TABLE IF EXISTS wind_instruments")
    print("Đã xóa bảng wind_instruments cũ (nếu có)")
    
    # Tạo bảng mới
    cursor.execute("""
        CREATE TABLE wind_instruments (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255),
            instrument_name VARCHAR(50),
            fundamental_freq FLOAT,
            mfcc_1 FLOAT, mfcc_2 FLOAT, mfcc_3 FLOAT, mfcc_4 FLOAT, mfcc_5 FLOAT,
            mfcc_6 FLOAT, mfcc_7 FLOAT, mfcc_8 FLOAT, mfcc_9 FLOAT, mfcc_10 FLOAT,
            mfcc_11 FLOAT, mfcc_12 FLOAT, mfcc_13 FLOAT,
            spectral_centroid FLOAT,
            zcr FLOAT,
            energy FLOAT
        )
    """)
    conn.commit()
    conn.close()
    print(status)

def insert_data():
    df = pd.read_csv(TRAIN_CSV)
    conn, status = get_db_connection()
    if conn is None:
        print(status)
        return
    
    cursor = conn.cursor()
    for _, row in df.iterrows():
        query = """
            INSERT INTO wind_instruments (filename, instrument_name, fundamental_freq, mfcc_1, mfcc_2, mfcc_3, mfcc_4, mfcc_5, mfcc_6, mfcc_7, mfcc_8, mfcc_9, mfcc_10, mfcc_11, mfcc_12, mfcc_13, spectral_centroid, zcr, energy)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            row["filename"], row["instrument_name"], row["fundamental_freq"],
            row["mfcc_1"], row["mfcc_2"], row["mfcc_3"], row["mfcc_4"], row["mfcc_5"],
            row["mfcc_6"], row["mfcc_7"], row["mfcc_8"], row["mfcc_9"], row["mfcc_10"],
            row["mfcc_11"], row["mfcc_12"], row["mfcc_13"],
            row["spectral_centroid"], row["zcr"], row["energy"]
        )
        cursor.execute(query, values)
    
    conn.commit()
    conn.close()
    print(status)
    print(f"Đã chèn {len(df)} bản ghi vào MySQL.")

if __name__ == "__main__":
    create_table()
    insert_data()