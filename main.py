"""Qwen3-ASR 桌面应用入口。"""

from gui import AsrGui


def main() -> None:
    app = AsrGui()
    app.run()


if __name__ == "__main__":
    main()
