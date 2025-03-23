import sqlite3
import time
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

# Khởi tạo cơ sở dữ liệu
def init_db():
    conn = sqlite3.connect("celebrity_cache.db")  # Tạo file DB
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS celebrity_cache (
            celeb_name TEXT PRIMARY KEY,
            result TEXT,
            timestamp REAL
        )
    """)
    conn.commit()
    conn.close()

# Kiểm tra và lấy dữ liệu từ DB
def get_from_db(celeb_name, max_age=86400):  # max_age: 24 giờ
    conn = sqlite3.connect("celebrity_cache.db")
    cursor = conn.cursor()
    cursor.execute("SELECT result, timestamp FROM celebrity_cache WHERE celeb_name = ?", (celeb_name,))
    row = cursor.fetchone()
    conn.close()
    if row:
        result, timestamp = row
        if time.time() - timestamp < max_age:  # Kiểm tra dữ liệu còn "tươi"
            return result
    return None

# Lưu dữ liệu vào DB
def save_to_db(celeb_name, result):
    conn = sqlite3.connect("celebrity_cache.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO celebrity_cache (celeb_name, result, timestamp)
        VALUES (?, ?, ?)
    """, (celeb_name, result, time.time()))
    conn.commit()
    conn.close()

# Hàm tìm kiếm với DB
def gg_tool(celeb_name, max_results=5):
    # Kiểm tra trong DB trước
    cached_result = get_from_db(celeb_name)
    if cached_result:
        return cached_result

    # Nếu không có trong DB, gọi API
    try:
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = tavily_client.search(query=celeb_name, max_results=max_results)
        if not response["results"]:
            return f"No information found for {celeb_name}"
        text = "\n\n".join(("title: " + r["title"] + "\ncontent: " + r["content"]) for r in response["results"])
        # Lưu vào DB
        save_to_db(celeb_name, text)
        return text
    except Exception as e:
        return f"Search error: {str(e)}"

# Khởi tạo DB khi chạy chương trình
if __name__ == "__main__":
    init_db()
    # Ví dụ sử dụng
    print(gg_tool("Tom Cruise"))  # Lần đầu: gọi API, lưu DB
    print(gg_tool("Tom Cruise"))  # Lần hai: lấy từ DB