import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    "host": "localhost",
    "user": "root",       # Thay bằng thông tin của bạn
    "password": "",
    "database": "audio_db"
}

def get_db_connection():
    """
    Kết nối đến cơ sở dữ liệu MySQL và trả về trạng thái cùng đối tượng kết nối.
    Returns:
        tuple: (connection, status_message)
        - connection: Đối tượng kết nối MySQL (hoặc None nếu thất bại)
        - status_message: Chuỗi mô tả trạng thái kết nối
    """
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection, "Kết nối cơ sở dữ liệu thành công"
    except Error as e:
        return None, f"Lỗi kết nối cơ sở dữ liệu: {str(e)}"
    return None, "Không thể kết nối cơ sở dữ liệu (lỗi không xác định)"