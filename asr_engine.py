import threading
from typing import List, Optional

import torch
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.qwen3_asr import ASRTranscription

from config import (
    DEFAULT_ALIGNER_MODEL_PATH,
    DEFAULT_ASR_MODEL_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_MAX_NEW_TOKENS,
)
from main_logger import get_logger

_logger = get_logger(__name__)


class AsrEngine:
    """Qwen3-ASR 引擎，封装模型加载与转写推理。"""

    def __init__(self) -> None:
        self._model: Optional[Qwen3ASRModel] = None
        self._model_path: str = ""
        self._aligner_path: str = ""
        self._lock = threading.Lock()
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载。"""
        return self._is_loaded

    @property
    def model_path(self) -> str:
        """当前加载的 ASR 模型路径。"""
        return self._model_path

    @property
    def aligner_path(self) -> str:
        """当前加载的 ForcedAligner 路径。"""
        return self._aligner_path

    def load_model(
        self,
        model_path: str = DEFAULT_ASR_MODEL_PATH,
        forced_aligner_path: str = DEFAULT_ALIGNER_MODEL_PATH,
        device: str = DEFAULT_DEVICE,
        dtype: torch.dtype = DEFAULT_DTYPE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        """
        加载 ASR 模型和可选的 ForcedAligner。

        Args:
            model_path: ASR 模型路径或 HuggingFace 模型名
            forced_aligner_path: ForcedAligner 模型路径，空字符串表示不使用
            device: 推理设备
            dtype: 推理精度
            batch_size: 最大推理批大小
            max_new_tokens: 最大生成 token 数
        """
        with self._lock:
            if self._is_loaded:
                self._unload_model_unlocked()

            _logger.info("正在加载 ASR 模型: %s", model_path)

            model_kwargs = {
                "dtype": dtype,
                "device_map": device,
                "max_inference_batch_size": batch_size,
                "max_new_tokens": max_new_tokens,
            }

            if forced_aligner_path:
                _logger.info("正在加载 ForcedAligner: %s", forced_aligner_path)
                model_kwargs["forced_aligner"] = forced_aligner_path
                model_kwargs["forced_aligner_kwargs"] = {
                    "dtype": dtype,
                    "device_map": device,
                }

            self._model = Qwen3ASRModel.from_pretrained(model_path, **model_kwargs)
            self._model_path = model_path
            self._aligner_path = forced_aligner_path
            self._is_loaded = True

            _logger.info("模型加载完成 (设备: %s, 精度: %s)", device, str(dtype))

    def _unload_model_unlocked(self) -> None:
        """卸载模型（需在持有锁时调用）。"""
        if self._model is not None:
            del self._model
            self._model = None
        self._is_loaded = False
        self._model_path = ""
        self._aligner_path = ""

    def unload_model(self) -> None:
        """卸载模型并释放显存。"""
        with self._lock:
            _logger.info("正在卸载模型...")
            self._unload_model_unlocked()
            _logger.info("模型已卸载")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        return_time_stamps: bool = False,
    ) -> List[ASRTranscription]:
        """
        对音频文件执行语音识别。

        Args:
            audio_path: 音频文件路径
            language: 语言名称，None 表示自动检测
            return_time_stamps: 是否返回时间戳

        Returns:
            ASRTranscription 列表

        Raises:
            RuntimeError: 模型未加载
        """
        with self._lock:
            if not self._is_loaded or self._model is None:
                raise RuntimeError("模型未加载，请先调用 load_model()")

            _logger.info(
                "开始识别: %s (语言: %s, 时间戳: %s)",
                audio_path,
                language or "自动检测",
                return_time_stamps,
            )

            results = self._model.transcribe(
                audio=audio_path,
                language=language,
                return_time_stamps=return_time_stamps,
            )

            if results:
                _logger.info(
                    "识别完成，检测语言: %s，文本长度: %d",
                    results[0].language,
                    len(results[0].text),
                )
            else:
                _logger.warning("识别完成但无结果")

            return results
