import logging
import queue
import threading
from typing import List


class GuiLogHandler(logging.Handler):
    """将日志记录写入线程安全队列，供 GUI 轮询读取。"""

    def __init__(self, max_records: int = 500) -> None:
        super().__init__()
        self._queue: queue.Queue = queue.Queue()
        self._max_records = max_records
        self.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self._queue.put(msg)

    def drain(self) -> List[str]:
        """取出队列中所有待处理的日志记录。"""
        records: List[str] = []
        while not self._queue.empty():
            try:
                records.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return records


# 全局单例
_handler: GuiLogHandler = None
_lock = threading.Lock()


def _init_handler() -> GuiLogHandler:
    global _handler
    with _lock:
        if _handler is None:
            _handler = GuiLogHandler()
    return _handler


def get_handler() -> GuiLogHandler:
    """获取全局 GuiLogHandler 实例。"""
    if _handler is None:
        return _init_handler()
    return _handler


def get_logger(name: str) -> logging.Logger:
    """获取配置好的 logger 实例。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(get_handler())
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
