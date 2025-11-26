import os
import time
from datetime import datetime
import json
import base64
import mimetypes
from openai import OpenAI
from extract_text import extract_text

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="any")


def save_json_output(json_string, image_path):
    """
    Lưu chuỗi JSON vào file .json trong thư mục extract_json.
    Tên file = <tên ảnh>_<timestamp>.json
    """

    # Tạo thư mục nếu chưa tồn tại
    output_dir = "extract_json"
    os.makedirs(output_dir, exist_ok=True)

    # Tên file gốc
    base = os.path.basename(image_path)
    name, _ = os.path.splitext(base)

    # Timestamp để không trùng
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Đường dẫn file JSON
    json_path = os.path.join(output_dir, f"{name}_{timestamp}.json")

    # Chuyển chuỗi JSON thành object để kiểm tra tính hợp lệ
    try:
        data = json.loads(json_string)
    except Exception:
        # Nếu model trả về JSON lỗi, vẫn lưu chuỗi thô
        json_path = os.path.join(output_dir, f"{name}_{timestamp}_RAW.txt")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)
        return json_path

    # Lưu JSON đẹp + UTF-8
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return json_path


def get_messages(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    image_payload = {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{base64_image}"}
    }

    text_payload = {
        "type": "text",
        "text": (
            "Bạn là assistant chuyên trích xuất hóa đơn giá trị gia tăng (gtgt) Việt Nam."
            "Bạn chỉ được trả về đúng 1 chuỗi JSON duy nhất, không thêm chữ khác.\n"
            "Không có thông tin ghi \"N/A\".\n"
            "Trả đúng cấu trúc:\n"
            "{\n"
            "  \"ngay_hoa_don\": \"\",\n"
            "  \"ma_co_quan_thue\": \"\",\n"
            "  \"nguoi_ban\": {\n"
            "    \"ten_don_vi\": \"\",\n"
            "    \"ma_so_thue\": \"\",\n"
            "    \"dia_chi\": \"\",\n"
            "    \"dien_thoai\": \"\",\n"
            "    \"so_tai_khoan\": \"\"\n"
            "  },\n"
            "  \"nguoi_mua\": {\n"
            "    \"ten_nguoi_mua\": \"\",\n"
            "    \"ten_don_vi\": \"\",\n"
            "    \"ma_so_thue\": \"\",\n"
            "    \"dia_chi\": \"\",\n"
            "    \"so_tai_khoan\": \"\"\n"
            "  },\n"
            "  \"hinh_thuc_thanh_toan\": \"\",\n"
            "  \"mat_hang\": [\n"
            "    {\n"
            "      \"ten_hang\": \"\",\n"
            "      \"don_vi_tinh\": \"\",\n"
            "      \"so_luong\": \"\",\n"
            "      \"don_gia\": \"\",\n"
            "      \"thanh_tien\": \"\",\n"
            "      \"thue_suat\": \"\"\n"
            "    }\n"
            "  ],\n"
            "  \"tong_tien_thanh_toan\": \"\"\n"
            "}"
        )
    }


    return [
        {"role": "user", "content": [image_payload, text_payload]}
    ]


def generate(messages):
    start = time.time()

    response = client.chat.completions.create(
        model="qwen3-vl",
        messages=messages,
        temperature=0.0,
        top_p=0.1,
        extra_body={"max_new_tokens": 1028}
    )

    print(f"Inference time: {time.time() - start:.2f}s")

    if not response.choices:
        raise RuntimeError("Model không trả về kết quả!")

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    print("=== OCR HÓA ĐƠN VIỆT NAM – Qwen3-VL-4B + vLLM ===")

    while True:
        path = input("\nĐường dẫn ảnh (exit, quit, e, q để thoát): ").strip()
        if path.lower() in ["exit", "quit", "e", "q"]:
            print("Tạm biệt!")
            break

        if not path:
            print("Nhập đường dẫn ảnh!")
            continue

        print(f"Đang xử lý: {path}")

        try:
            msgs = get_messages(path)
            raw = generate(msgs)
            clean = extract_text(raw)

            print("\n====== KẾT QUẢ TRÍCH XUẤT ======")
            print(clean)
            print("=" * 60 + "\n")

            # Lưu file JSON
            saved_path = save_json_output(clean, path)
            print(f"\n➡ File JSON đã lưu tại: {saved_path}")
            print("=" * 60 + "\n")

        except Exception as e:
            print(f"Lỗi: {e}")
            import traceback
            traceback.print_exc()
