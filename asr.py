import torch
import re
from typing import List, Tuple
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.qwen3_forced_aligner import ForcedAlignItem

PUNCTUATION_PATTERN = re.compile(r"[、。！？，．,!\?\.\s]")

def _build_position_map(text: str, items: List[ForcedAlignItem]) -> List[int]:
    """
    建立去标点文本中每个字符在原文中的位置映射。

    同时验证去标点文本与对齐项拼接文本一致。

    Returns:
        clean_to_orig: 列表，长度为去标点后的字符数，
                       clean_to_orig[i] = 该字符在原文中的位置
    """
    clean_to_orig: List[int] = []
    for i, ch in enumerate(text):
        if not PUNCTUATION_PATTERN.match(ch):
            clean_to_orig.append(i)

    clean_text_str = "".join(text[p] for p in clean_to_orig)
    aligned_text = "".join(it.text for it in items)

    if clean_text_str != aligned_text:
        raise ValueError(
            f"去标点文本与对齐文本不匹配。\n"
            f"  原文(去标点): {clean_text_str}\n"
            f"  对齐拼接:     {aligned_text}"
        )

    return clean_to_orig

def _segment_items_by_gap(
    items: List[ForcedAlignItem],
    gap_threshold: float = 0.4,
    max_duration: float = 10.0,
    ) -> List[Tuple[int, int]]:
    """
    按时间间隔和最大时长将对齐项分割成段。

    Args:
        items: 对齐项列表
        gap_threshold: 两项之间超过此秒数则断句
        max_duration: 单段最大时长，超过时强制分割

    Returns:
        [(start_idx, end_idx), ...] 每段在 items 中的索引范围
    """
    segments: List[Tuple[int, int]] = []
    seg_start = 0

    for i in range(len(items)):
        is_last = i == len(items) - 1

        # 检查到下一项的时间间隔
        large_gap = False
        if not is_last:
            gap = items[i + 1].start_time - items[i].end_time
            large_gap = gap > gap_threshold

        # 检查当前段是否超过最大时长
        cur_duration = items[i].end_time - items[seg_start].start_time
        too_long = cur_duration >= max_duration

        if large_gap or too_long or is_last:
            segments.append((seg_start, i))
            seg_start = i + 1

    return segments

def _extract_original_text(
    text: str,
    seg_s: int,
    seg_e: int,
    items: List[ForcedAlignItem],
    clean_to_orig: List[int],
    ) -> str:
    """
    提取对齐项段落在原文中对应的文本（含标点）。

    逻辑：计算段落在 clean_text 中的字符范围 → 映射回原文位置 → 截取原文子串。
    若段尾后紧跟句末标点（。！？），一并纳入。
    """
    # 计算每项在 clean_text 中的字符范围
    char_pos = 0
    item_starts: List[int] = []
    for item in items:
        item_starts.append(char_pos)
        char_pos += len(item.text)

    clean_start = item_starts[seg_s]
    item_e = items[seg_e]
    clean_end = item_starts[seg_e] + len(item_e.text) - 1

    orig_start = clean_to_orig[clean_start]
    orig_end = clean_to_orig[clean_end]

    # 基础文本：从原文截取（含非句末标点，如顿号逗号）
    display = text[orig_start : orig_end + 1]

    # 若段尾后紧跟句末标点（。！？），一并纳入
    if orig_end + 1 < len(text):
        ch = text[orig_end + 1]
        if ch in "。！？":
            display += ch

    return display.strip()

def _format_timestamp(seconds: float) -> str:
    """将秒数转换为 SRT 时间格式 HH:MM:SS,mmm。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        s += 1
        ms -= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def generate_srt(
    text: str,
    items: List[ForcedAlignItem],
    gap_threshold: float = 0.4,
    max_duration: float = 10.0,
    ) -> str:
    """
    根据原文和对齐项生成 SRT 格式字幕。

    Args:
        text: ASR 识别的完整文本（含标点）
        items: 强制对齐结果列表
        gap_threshold: 两项时间间隔超过此秒数则断句（默认 0.4s）
        max_duration: 单条字幕最大时长，超时强制分割（默认 10s）

    Returns:
        SRT 格式字符串
    """
    if not items:
        return ""

    clean_to_orig = _build_position_map(text, items)
    segments = _segment_items_by_gap(items, gap_threshold, max_duration)

    srt_lines: List[str] = []
    for seq, (s, e) in enumerate(segments, 1):
        start_time = _format_timestamp(items[s].start_time)
        end_time = _format_timestamp(items[e].end_time)
        display_text = _extract_original_text(text, s, e, items, clean_to_orig)

        srt_lines.append(str(seq))
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(display_text)
        srt_lines.append("")

    return "\n".join(srt_lines)

def qwen_transcribe(
        model_path: str = "./Qwen3-ASR-1.7B",
        forced_aligner_path: str = "./Qwen3-ForcedAligner-0.6B",
        audio_path: str = "audio.wav",
        language: str = "Japanese",
        return_time_stamps: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0"
    ) -> List[str]:
    model = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device,
        # attn_implementation="flash_attention_2",
        max_inference_batch_size=24, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
        max_new_tokens=10240, # Maximum number of tokens to generate. Set a larger value for long audio input.
        forced_aligner=forced_aligner_path,
            dtype=dtype,
            device_map=device,
            # attn_implementation="flash_attention_2",
        )

    results = model.transcribe(
        audio=audio_path,
        language=language, # can also be set to None for automatic language detection
        return_time_stamps=return_time_stamps,
    )

    return [generate_srt(r.text, r.time_stamps) for r in results]

if __name__ == "__main__":
    srt_list = qwen_transcribe(
        model_path = "./Qwen3-ASR-1.7B",
        forced_aligner_path = "./Qwen3-ForcedAligner-0.6B",
        audio_path = "audio.wav",
        language = "Japanese",
        return_time_stamps = True,
        dtype = torch.bfloat16,
        device = "cuda:0"
    )