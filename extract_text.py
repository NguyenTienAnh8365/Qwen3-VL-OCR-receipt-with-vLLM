def extract_text(output):
    """
    Trích xuất text sạch từ mọi loại output của VLM 2025
    Hỗ trợ: Unsloth, Transformers, vLLM, OpenAI, Grok, Claude, Ollama, llama.cpp, ...
    """
    # Case 1: Đã là string → trả luôn
    if isinstance(output, str):
        return output.strip()

    # Case 2: Object có .text hoặc .content (vLLM, OpenAI, Grok, Claude, Gemini)
    if hasattr(output, "text"):
        return (getattr(output, "text") or "").strip()
    if hasattr(output, "content"):
        return (getattr(output, "content") or "").strip()

    # Case 3: Dict kiểu OpenAI API / Ollama / llama.cpp
    if isinstance(output, dict):
        # OpenAI style
        if "choices" in output and len(output["choices"]) > 0:
            msg = output["choices"][0].get("message") or output["choices"][0]
            return (msg.get("content") or msg.get("text") or "").strip()
        # Ollama / llama.cpp style
        if "choices" in output and "text" in output["choices"][0]:
            return output["choices"][0]["text"].strip()
        # Simple dict có key "content" hoặc "text"
        return (output.get("content") or output.get("text") or str(output)).strip()

    # Case 4: List các object (vLLM batch mode)
    if isinstance(output, (list, tuple)) and len(output) > 0:
        return extract_text(output[0])  # Đệ quy lấy phần tử đầu

    # Case 5: vLLM GenerationOutput object
    if hasattr(output, "outputs") and len(output.outputs) > 0:
        return extract_text(output.outputs[0])

    # Case cuối: Không biết kiểu gì → ép string
    return str(output).strip()