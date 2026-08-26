from contextlib import asynccontextmanager
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

check_hallucination_fn = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global check_hallucination_fn
    print("Đang nạp mô hình từ chay.py...")
    from chay import check_hallucination
    check_hallucination_fn = check_hallucination
    print("Hệ thống đã sẵn sàng!")
    yield

app = FastAPI(title="LLM Hallucination Detection API", lifespan=lifespan)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CheckRequest(BaseModel):
    answer: str

# 1. Route API xử lý kiểm tra (Đặt trên cùng để không bị xung đột route)
@app.post("/api/check")
def check(req: CheckRequest):
    if not req.answer.strip():
        raise HTTPException(status_code=400, detail="Nội dung không được để trống!")
    
    if check_hallucination_fn is None:
        raise HTTPException(status_code=500, detail="Mô hình chưa sẵn sàng!")

    # Trả về kết quả từ hàm trong chay.py
    return check_hallucination_fn(req.answer)

# 2. Trang chủ trả về file HTML
@app.get("/")
def get_ui():
    html_file = "GiaoDien.html"
    if not os.path.exists(html_file):
        raise HTTPException(status_code=404, detail="Không tìm thấy file GiaoDien.html")
    return FileResponse(html_file)

# 3. Phục vụ các file tĩnh CSS/JS thông qua tiền tố /static
app.mount("/static", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    print("\n==================================================")
    print(" Bấm Ctrl + Click vào link dưới đây để mở giao diện:")
    print(" http://127.0.0.1:8000")
    print("==================================================\n")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)