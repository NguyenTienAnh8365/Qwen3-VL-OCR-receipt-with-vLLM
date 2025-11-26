import subprocess
import os

os.environ["VLLM_WORKER_MULTIPROCESS_METHOD"] = "spawn"

cmd = [
	"vllm", "serve", "unsloth/Qwen3-VL-4B-Instruct",
	"--dtype", "float16",
	"--trust-remote-code",
	"--max-model-len", "4096",
	"--gpu-memory-utilization", "0.8", # limit memory
	"--tensor-parallel-size", "1",
	"--port", "8000",
	"--served-model-name", "qwen3-vl",
	"--limit-mm-per-prompt", '{"image":2}', # limit images
	"--enforce-eager",
	"--max-num-batched-tokens", "2048", # bacth image tokens
	"--max-num-seqs", "2", # số request cùng lúc
]

print("Running vLLM server (Qwen3-VL-4B)...")
subprocess.run(cmd)