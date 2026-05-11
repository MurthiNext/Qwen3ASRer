import logging

import torch
from qwen_asr.inference.utils import SUPPORTED_LANGUAGES

# 模型默认路径
DEFAULT_ASR_MODEL_PATH = "./Qwen3-ASR-1.7B"
DEFAULT_ALIGNER_MODEL_PATH = "./Qwen3-ForcedAligner-0.6B"

# 推理默认参数
DEFAULT_DEVICE = "cuda:0"
DEFAULT_DTYPE = torch.bfloat16
DEFAULT_BATCH_SIZE = 24
DEFAULT_MAX_NEW_TOKENS = 10240

# SRT/LRC 分段默认参数
DEFAULT_GAP_THRESHOLD = 0.4
DEFAULT_MAX_DURATION = 10.0

# 输出格式选项
OUTPUT_FORMATS = ["srt", "lrc"]

# 日志配置
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

# 支持的语言列表
LANGUAGES = SUPPORTED_LANGUAGES

# 语言选项（含"自动检测"）
LANGUAGE_OPTIONS = ["自动检测"] + LANGUAGES
