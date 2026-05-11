import re
from typing import List, Tuple

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

        large_gap = False
        if not is_last:
            gap = items[i + 1].start_time - items[i].end_time
            large_gap = gap > gap_threshold

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

    display = text[orig_start: orig_end + 1]

    if orig_end + 1 < len(text):
        ch = text[orig_end + 1]
        if ch in "。！？":
            display += ch

    return display.strip()


def _format_timestamp_srt(seconds: float) -> str:
    """将秒数转换为 SRT 时间格式 HH:MM:SS,mmm。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        s += 1
        ms -= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_lrc(seconds: float) -> str:
    """将秒数转换为 LRC 时间格式 [mm:ss.xx]。"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        s += 1
        cs -= 100
    return f"[{m:02d}:{s:02d}.{cs:02d}]"


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
        start_time = _format_timestamp_srt(items[s].start_time)
        end_time = _format_timestamp_srt(items[e].end_time)
        display_text = _extract_original_text(text, s, e, items, clean_to_orig)

        srt_lines.append(str(seq))
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(display_text)
        srt_lines.append("")

    return "\n".join(srt_lines)


def generate_lrc(
    items: List[ForcedAlignItem],
    gap_threshold: float = 0.4,
    max_duration: float = 10.0,
) -> str:
    """
    根据对齐项生成 LRC 格式歌词。

    使用 items 的文本和时间戳，每行格式为 [mm:ss.xx]文本。

    Args:
        items: 强制对齐结果列表
        gap_threshold: 两项时间间隔超过此秒数则断句（默认 0.4s）
        max_duration: 单条歌词最大时长，超时强制分割（默认 10s）

    Returns:
        LRC 格式字符串
    """
    if not items:
        return ""

    segments = _segment_items_by_gap(items, gap_threshold, max_duration)

    lrc_lines: List[str] = []
    for s, e in segments:
        timestamp = _format_timestamp_lrc(items[s].start_time)
        # 拼接段落文本
        seg_text = "".join(it.text for it in items[s: e + 1])
        lrc_lines.append(f"{timestamp}{seg_text}")

    return "\n".join(lrc_lines)


def export_subtitle(
    text: str,
    items: List[ForcedAlignItem],
    fmt: str,
    output_path: str,
    gap_threshold: float = 0.4,
    max_duration: float = 10.0,
) -> None:
    """
    统一导出字幕/歌词文件。

    Args:
        text: ASR 识别的完整文本（含标点，LRC 格式不需要但 SRT 需要）
        items: 强制对齐结果列表
        fmt: 输出格式，"srt" 或 "lrc"
        output_path: 输出文件路径
        gap_threshold: 断句间隔阈值
        max_duration: 单段最大时长
    """
    if fmt == "srt":
        content = generate_srt(text, items, gap_threshold, max_duration)
    elif fmt == "lrc":
        content = generate_lrc(items, gap_threshold, max_duration)
    else:
        raise ValueError(f"不支持的格式: {fmt}，请使用 'srt' 或 'lrc'")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
