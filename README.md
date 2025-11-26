# Vietnamese Invoice OCR with Qwen3-VL-4B

Extract Vietnamese VAT invoices (Hóa đơn giá trị gia tăng) using Qwen3-VL-4B model with vLLM inference server.

## Features

- 🚀 Fast inference with vLLM backend
- 📄 Automatic JSON structure extraction from invoice images
- 🇻🇳 Optimized for Vietnamese VAT invoices
- 💾 Auto-save extracted data to JSON files
- 🔄 Interactive CLI interface

## Requirements

- NVIDIA GPU with CUDA support (Driver 580+, CUDA 12.1+)
- Python 3.12
- 12GB+ GPU memory recommended

## Installation

### 1. Install NVIDIA Driver (if needed)

```bash
# Check if driver 580 is already installed
sudo apt update
sudo apt install nvidia-driver-580 nvidia-utils-580 -y
sudo reboot
# Verify CUDA version 13.0 is available in 2025
```

### 2. Install Miniconda (if not installed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Create Environment and Install Dependencies

```bash
# Create Python 3.12 environment
conda create -n qwen3 python=3.12 -y
conda activate qwen3

# Install CUDA toolkit (use 12.1 for fast and stable support with vLLM)
conda install -c nvidia cuda-toolkit=12.1 -y

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM (2025 version with optimized Qwen3-VL support)
pip install vllm==0.7.1

# Install OpenAI client + Pillow for image processing
pip install openai pillow
```

## Project Structure

```
.
├── server.py           # vLLM server launcher
├── main.py            # Main OCR script
├── extract_text.py    # Text extraction utility
├── extract_json/      # Output directory for extracted JSON files
└── README.md
```

## Usage

### Step 1: Start vLLM Server

Open a terminal and run:

```bash
conda activate qwen3
python server.py
```

Wait for the server to load the model (typically 1-2 minutes). You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 2: Run OCR Extraction

In another terminal:

```bash
conda activate qwen3
python main.py
```

Enter the path to your invoice image when prompted:

```
=== OCR HÓA ĐƠN VIỆT NAM – Qwen3-VL-4B + vLLM ===

Đường dẫn ảnh (exit, quit, e, q để thoát): /path/to/invoice.jpg
```

### Output

The script will:
1. Process the invoice image
2. Extract structured data
3. Display results in terminal
4. Save JSON file to `extract_json/` directory

Example output structure:

```json
{
  "ngay_hoa_don": "15/03/2025",
  "ma_co_quan_thue": "123456789",
  "nguoi_ban": {
    "ten_don_vi": "CÔNG TY ABC",
    "ma_so_thue": "0123456789",
    "dia_chi": "123 Đường XYZ, Hà Nội",
    "dien_thoai": "0243123456",
    "so_tai_khoan": "12345678901234"
  },
  "nguoi_mua": {
    "ten_nguoi_mua": "Nguyễn Văn A",
    "ten_don_vi": "CÔNG TY XYZ",
    "ma_so_thue": "9876543210",
    "dia_chi": "456 Đường ABC, TP.HCM",
    "so_tai_khoan": "98765432109876"
  },
  "hinh_thuc_thanh_toan": "Chuyển khoản",
  "mat_hang": [
    {
      "ten_hang": "Sản phẩm A",
      "don_vi_tinh": "cái",
      "so_luong": "10",
      "don_gia": "100,000",
      "thanh_tien": "1,000,000",
      "thue_suat": "10%"
    }
  ],
  "tong_tien_thanh_toan": "1,100,000"
}
```

## Configuration

### Server Parameters (server.py)

- `--max-model-len`: Maximum sequence length (default: 8192)
- `--gpu-memory-utilization`: GPU memory usage ratio (default: 0.8)
- `--max-num-seqs`: Max concurrent requests (default: 2)
- `--max-num-batched-tokens`: Batch size for image tokens (default: 2048)

### Model Parameters (main.py)

- `temperature`: 0.0 (deterministic output)
- `top_p`: 0.1 (focused sampling)
- `max_new_tokens`: 1028 (maximum output length)

## Troubleshooting

### CUDA Out of Memory

Reduce GPU memory utilization:
```python
# In server.py
"--gpu-memory-utilization", "0.6",  # Reduce from 0.8 to 0.6
```

### Slow Inference

- Ensure CUDA 12.1 is properly installed
- Check GPU utilization with `nvidia-smi`
- Reduce `--max-model-len` if needed

### Invalid JSON Output

The script automatically handles malformed JSON by saving raw output to a `.txt` file with `_RAW` suffix.

## Performance

- **Model**: Qwen3-VL-4B (4 billion parameters)
- **Inference time**: ~20-40 seconds per invoice (depending on size, image complexity and GPU)
- **GPU memory**: ~12-16GB VRAM

## License

MIT License

## Credits

- Model: [Qwen3-VL-4B-Instruct](https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct) by Unsloth
- Inference: [vLLM](https://github.com/vllm-project/vllm)

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
