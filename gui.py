import os
import threading
from tkinter import filedialog, messagebox

import customtkinter as ctk
import librosa
import torch

from asr_engine import AsrEngine
from config import (
    DEFAULT_ALIGNER_MODEL_PATH,
    DEFAULT_ASR_MODEL_PATH,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_GAP_THRESHOLD,
    DEFAULT_MAX_DURATION,
    LANGUAGE_OPTIONS,
    OUTPUT_FORMATS,
)
from main_logger import get_handler, get_logger
from subtitle_generator import export_subtitle

# 布局常量
LABEL_WIDTH = 100
ROW_PADY = 3
SECTION_PADX = 12
SECTION_PADY = (6, 2)

_logger = get_logger(__name__)


class AsrGui:
    """Qwen3-ASR 桌面应用主界面。"""

    def __init__(self) -> None:
        self._engine = AsrEngine()
        self._results = None
        self._audio_path = ""
        self._audio_duration = 0.0

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._root = ctk.CTk()
        self._root.title("Qwen3-ASR 语音识别")
        self._root.geometry("1000x700")
        self._root.minsize(880, 600)

        self._define_components()
        self._layout_components()
        self._start_log_polling()

    # 组件定义 (只创建，不放置)

    def _define_components(self) -> None:
        self._define_main_panels()
        self._define_model_section()
        self._define_audio_section()
        self._define_recognition_section()
        self._define_output_section()
        self._define_action_section()
        self._define_log_panel()
        self._define_status_bar()

    def _define_main_panels(self) -> None:
        self._left_panel = ctk.CTkFrame(self._root)
        self._right_panel = ctk.CTkFrame(self._root)

        self._title_label = ctk.CTkLabel(
            self._left_panel,
            text="Qwen3-ASR 语音识别",
            font=ctk.CTkFont(size=20, weight="bold"),
        )

    def _define_model_section(self) -> None:
        self._model_frame = ctk.CTkFrame(self._left_panel)
        self._model_header = ctk.CTkLabel(
            self._model_frame,
            text="模型设置",
            font=ctk.CTkFont(size=13, weight="bold"),
        )

        # ASR 模型路径
        self._model_path_var = ctk.StringVar(value=DEFAULT_ASR_MODEL_PATH)
        self._model_path_label = ctk.CTkLabel(
            self._model_frame, text="ASR 模型:", width=LABEL_WIDTH, anchor="e"
        )
        self._model_path_entry = ctk.CTkEntry(
            self._model_frame, textvariable=self._model_path_var
        )
        self._model_browse_btn = ctk.CTkButton(
            self._model_frame, text="浏览...", width=60, command=self._browse_model
        )

        # Aligner 路径
        self._aligner_path_var = ctk.StringVar(value=DEFAULT_ALIGNER_MODEL_PATH)
        self._aligner_path_label = ctk.CTkLabel(
            self._model_frame, text="Aligner:", width=LABEL_WIDTH, anchor="e"
        )
        self._aligner_path_entry = ctk.CTkEntry(
            self._model_frame, textvariable=self._aligner_path_var
        )
        self._aligner_browse_btn = ctk.CTkButton(
            self._model_frame, text="浏览...", width=60, command=self._browse_aligner
        )

        # 设备 + 精度
        self._device_label = ctk.CTkLabel(
            self._model_frame, text="设备:", width=LABEL_WIDTH, anchor="e"
        )
        self._device_var = ctk.StringVar(value=DEFAULT_DEVICE)
        self._device_menu = ctk.CTkOptionMenu(
            self._model_frame,
            values=["cuda:0", "cpu"],
            variable=self._device_var,
            width=110,
        )
        self._dtype_label = ctk.CTkLabel(
            self._model_frame, text="精度:", width=42, anchor="e"
        )
        self._dtype_var = ctk.StringVar(value="bfloat16")
        self._dtype_menu = ctk.CTkOptionMenu(
            self._model_frame,
            values=["bfloat16", "float16", "float32"],
            variable=self._dtype_var,
            width=110,
        )

    def _define_audio_section(self) -> None:
        self._audio_frame = ctk.CTkFrame(self._left_panel)
        self._audio_header = ctk.CTkLabel(
            self._audio_frame,
            text="音频输入",
            font=ctk.CTkFont(size=13, weight="bold"),
        )

        self._audio_path_var = ctk.StringVar()
        self._audio_path_label = ctk.CTkLabel(
            self._audio_frame, text="音频文件:", width=LABEL_WIDTH, anchor="e"
        )
        self._audio_path_entry = ctk.CTkEntry(
            self._audio_frame, textvariable=self._audio_path_var
        )
        self._audio_browse_btn = ctk.CTkButton(
            self._audio_frame, text="浏览...", width=60, command=self._browse_audio
        )

        self._audio_clear_btn = ctk.CTkButton(
            self._audio_frame, text="清空", width=50, command=self._clear_audio
        )
        self._audio_info_label = ctk.CTkLabel(
            self._audio_frame, text="未选择音频文件", text_color="gray60"
        )

    def _define_recognition_section(self) -> None:
        self._recog_frame = ctk.CTkFrame(self._left_panel)
        self._recog_header = ctk.CTkLabel(
            self._recog_frame,
            text="识别设置",
            font=ctk.CTkFont(size=13, weight="bold"),
        )

        # 语言
        self._language_label = ctk.CTkLabel(
            self._recog_frame, text="语言:", width=LABEL_WIDTH, anchor="e"
        )
        self._language_var = ctk.StringVar(value="自动检测")
        self._language_menu = ctk.CTkOptionMenu(
            self._recog_frame,
            values=LANGUAGE_OPTIONS,
            variable=self._language_var,
            width=150,
        )

        # 时间戳
        self._timestamps_var = ctk.BooleanVar(value=True)
        self._timestamps_check = ctk.CTkCheckBox(
            self._recog_frame, text="启用时间戳", variable=self._timestamps_var
        )

        # 断句间隔
        self._gap_label = ctk.CTkLabel(
            self._recog_frame, text="断句间隔(s):", width=LABEL_WIDTH, anchor="e"
        )
        self._gap_var = ctk.StringVar(value=str(DEFAULT_GAP_THRESHOLD))
        self._gap_entry = ctk.CTkEntry(self._recog_frame, textvariable=self._gap_var, width=60)

        # 最大时长
        self._max_dur_label = ctk.CTkLabel(
            self._recog_frame, text="最大时长(s):", width=86, anchor="e"
        )
        self._max_dur_var = ctk.StringVar(value=str(DEFAULT_MAX_DURATION))
        self._max_dur_entry = ctk.CTkEntry(
            self._recog_frame, textvariable=self._max_dur_var, width=60
        )

    def _define_output_section(self) -> None:
        self._output_frame = ctk.CTkFrame(self._left_panel)
        self._output_header = ctk.CTkLabel(
            self._output_frame,
            text="输出设置",
            font=ctk.CTkFont(size=13, weight="bold"),
        )

        # 格式
        self._output_format_label = ctk.CTkLabel(
            self._output_frame, text="输出格式:", width=LABEL_WIDTH, anchor="e"
        )
        self._output_format_var = ctk.StringVar(value="srt")
        self._output_format_menu = ctk.CTkOptionMenu(
            self._output_frame,
            values=OUTPUT_FORMATS,
            variable=self._output_format_var,
            width=100,
        )

        # 文件名
        self._output_name_label = ctk.CTkLabel(
            self._output_frame, text="输出文件名:", width=LABEL_WIDTH, anchor="e"
        )
        self._output_name_var = ctk.StringVar()
        self._output_name_entry = ctk.CTkEntry(
            self._output_frame, textvariable=self._output_name_var
        )
        self._output_name_hint = ctk.CTkLabel(
            self._output_frame,
            text="留空则与音频文件同名，保存至音频同目录",
            text_color="gray60",
            font=ctk.CTkFont(size=11),
        )

    def _define_action_section(self) -> None:
        self._action_frame = ctk.CTkFrame(self._left_panel)

        self._start_btn = ctk.CTkButton(
            self._action_frame,
            text="开始识别",
            width=110,
            command=self._start_transcribe,
        )
        self._progress = ctk.CTkProgressBar(self._action_frame)

    def _define_log_panel(self) -> None:
        self._log_header = ctk.CTkLabel(
            self._right_panel,
            text="运行日志",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self._log_textbox = ctk.CTkTextbox(self._right_panel, wrap="word", state="disabled")
        self._log_clear_btn = ctk.CTkButton(
            self._right_panel,
            text="清空日志",
            width=80,
            command=self._clear_log,
        )

    def _define_status_bar(self) -> None:
        self._status_var = ctk.StringVar(value="就绪")
        self._status_label = ctk.CTkLabel(
            self._left_panel,
            textvariable=self._status_var,
            anchor="w",
            font=ctk.CTkFont(size=11),
            text_color="gray70",
        )

    # 布局 (只放置，不创建)

    def _layout_components(self) -> None:
        self._layout_main_panels()
        self._layout_model_section()
        self._layout_audio_section()
        self._layout_recognition_section()
        self._layout_output_section()
        self._layout_action_section()
        self._layout_status_bar()
        self._layout_log_panel()

        # 列权重配置
        self._root.grid_columnconfigure(0, weight=3)
        self._root.grid_columnconfigure(1, weight=2)
        self._root.grid_rowconfigure(0, weight=1)

    def _layout_main_panels(self) -> None:
        self._left_panel.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self._right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)

        self._title_label.pack(anchor="w", padx=SECTION_PADX, pady=(10, 6))

        self._left_panel.grid_columnconfigure(0, weight=1)
        self._right_panel.grid_columnconfigure(0, weight=1)
        self._right_panel.grid_rowconfigure(1, weight=1)

    def _layout_section_header(self, frame: ctk.CTkFrame, header: ctk.CTkLabel) -> None:
        header.grid(row=0, column=0, columnspan=4, sticky="w", padx=SECTION_PADX, pady=SECTION_PADY)

    def _layout_model_section(self) -> None:
        f = self._model_frame
        f.pack(fill="x", padx=8, pady=(0, 6))
        f.grid_columnconfigure(1, weight=1)

        self._layout_section_header(f, self._model_header)

        # ASR 模型
        self._model_path_label.grid(row=1, column=0, sticky="e", padx=(SECTION_PADX, 4), pady=ROW_PADY)
        self._model_path_entry.grid(row=1, column=1, sticky="ew", pady=ROW_PADY)
        self._model_browse_btn.grid(row=1, column=2, padx=(4, SECTION_PADX), pady=ROW_PADY)

        # Aligner
        self._aligner_path_label.grid(row=2, column=0, sticky="e", padx=(SECTION_PADX, 4), pady=ROW_PADY)
        self._aligner_path_entry.grid(row=2, column=1, sticky="ew", pady=ROW_PADY)
        self._aligner_browse_btn.grid(row=2, column=2, padx=(4, SECTION_PADX), pady=ROW_PADY)

        # 设备 + 精度
        self._device_label.grid(row=3, column=0, sticky="e", padx=(SECTION_PADX, 4), pady=ROW_PADY)
        self._device_menu.grid(row=3, column=1, sticky="w", pady=ROW_PADY)
        self._dtype_label.grid(row=3, column=1, sticky="e", padx=(0, 4), pady=ROW_PADY)
        self._dtype_menu.grid(row=3, column=2, padx=(4, SECTION_PADX), pady=ROW_PADY)

    def _layout_audio_section(self) -> None:
        f = self._audio_frame
        f.pack(fill="x", padx=8, pady=(0, 6))
        f.grid_columnconfigure(1, weight=1)

        self._layout_section_header(f, self._audio_header)

        self._audio_path_label.grid(row=1, column=0, sticky="e", padx=(SECTION_PADX, 4), pady=ROW_PADY)
        self._audio_path_entry.grid(row=1, column=1, sticky="ew", pady=ROW_PADY)
        self._audio_browse_btn.grid(row=1, column=2, padx=(4, SECTION_PADX), pady=ROW_PADY)

        self._audio_clear_btn.grid(row=2, column=1, sticky="w", padx=(0, 6), pady=(0, 4))
        self._audio_info_label.grid(row=2, column=1, sticky="w", padx=(60, 0), pady=(0, 4))

    def _layout_recognition_section(self) -> None:
        f = self._recog_frame
        f.pack(fill="x", padx=8, pady=(0, 6))
        f.grid_columnconfigure(1, weight=1)

        self._layout_section_header(f, self._recog_header)

        # 语言 + 时间戳
        self._language_label.grid(row=1, column=0, sticky="e", padx=(SECTION_PADX, 4), pady=ROW_PADY)
        self._language_menu.grid(row=1, column=1, sticky="w", pady=ROW_PADY)
        self._timestamps_check.grid(row=1, column=2, padx=(4, SECTION_PADX), pady=ROW_PADY)

        # 断句间隔 + 最大时长
        self._gap_label.grid(row=2, column=0, sticky="e", padx=(SECTION_PADX, 4), pady=ROW_PADY)
        self._gap_entry.grid(row=2, column=1, sticky="w", pady=ROW_PADY)
        self._max_dur_label.grid(row=2, column=1, sticky="e", padx=(0, 4), pady=ROW_PADY)
        self._max_dur_entry.grid(row=2, column=2, padx=(4, SECTION_PADX), pady=ROW_PADY)

    def _layout_output_section(self) -> None:
        f = self._output_frame
        f.pack(fill="x", padx=8, pady=(0, 6))
        f.grid_columnconfigure(1, weight=1)

        self._layout_section_header(f, self._output_header)

        self._output_format_label.grid(row=1, column=0, sticky="e", padx=(SECTION_PADX, 4), pady=ROW_PADY)
        self._output_format_menu.grid(row=1, column=1, sticky="w", pady=ROW_PADY)

        self._output_name_label.grid(row=2, column=0, sticky="e", padx=(SECTION_PADX, 4), pady=ROW_PADY)
        self._output_name_entry.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(0, SECTION_PADX), pady=ROW_PADY)

        self._output_name_hint.grid(
            row=3, column=1, columnspan=2, sticky="w", padx=(0, SECTION_PADX), pady=(0, 4)
        )

    def _layout_action_section(self) -> None:
        self._action_frame.pack(fill="x", padx=8, pady=(0, 4))

        self._start_btn.pack(side="left", padx=(SECTION_PADX, 12), pady=(8, 4))
        self._progress.pack(side="left", fill="x", expand=True, padx=(0, SECTION_PADX), pady=(8, 4))

    def _layout_status_bar(self) -> None:
        self._status_label.pack(fill="x", padx=SECTION_PADX, pady=(2, 8))

    def _layout_log_panel(self) -> None:
        self._log_header.pack(anchor="w", padx=10, pady=(10, 4))
        self._log_textbox.pack(fill="both", expand=True, padx=10, pady=(0, 4))
        self._log_clear_btn.pack(side="bottom", anchor="e", padx=10, pady=(0, 8))

    # 日志轮询

    def _start_log_polling(self) -> None:
        self._poll_log()

    def _poll_log(self) -> None:
        handler = get_handler()
        for msg in handler.drain():
            self._append_log(msg)
        self._root.after(200, self._poll_log)

    def _append_log(self, msg: str) -> None:
        self._log_textbox.configure(state="normal")
        self._log_textbox.insert("end", msg + "\n")
        self._log_textbox.see("end")
        self._log_textbox.configure(state="disabled")

    def _clear_log(self) -> None:
        self._log_textbox.configure(state="normal")
        self._log_textbox.delete("1.0", "end")
        self._log_textbox.configure(state="disabled")

    # 事件处理

    def _browse_model(self) -> None:
        path = filedialog.askdirectory(title="选择 ASR 模型目录")
        if path:
            self._model_path_var.set(path)

    def _browse_aligner(self) -> None:
        path = filedialog.askdirectory(title="选择 ForcedAligner 模型目录")
        if path:
            self._aligner_path_var.set(path)

    def _browse_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[
                ("音频文件", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("所有文件", "*.*"),
            ],
        )
        if path:
            self._set_audio_path(path)

    def _set_audio_path(self, path: str) -> None:
        self._audio_path = path
        self._audio_path_var.set(path)
        try:
            duration = librosa.get_duration(path=path)
            self._audio_duration = duration
            m, s = int(duration // 60), int(duration % 60)
            self._audio_info_label.configure(
                text=f"时长: {m:02d}:{s:02d}  |  {os.path.basename(path)}",
                text_color="white",
            )
        except Exception:
            self._audio_duration = 0.0
            self._audio_info_label.configure(
                text=f"无法读取音频信息: {os.path.basename(path)}",
                text_color="orange",
            )

    def _clear_audio(self) -> None:
        self._audio_path = ""
        self._audio_duration = 0.0
        self._audio_path_var.set("")
        self._audio_info_label.configure(text="未选择音频文件", text_color="gray60")

    def _start_transcribe(self) -> None:
        audio_path = self._audio_path_var.get().strip()
        if not audio_path or not os.path.isfile(audio_path):
            messagebox.showerror("错误", "请先选择有效的音频文件")
            return

        model_path = self._model_path_var.get().strip()
        if not model_path:
            messagebox.showerror("错误", "请设置 ASR 模型路径")
            return

        self._set_ui_state(running=True)

        thread = threading.Thread(target=self._run_transcribe, daemon=True)
        thread.start()

    def _run_transcribe(self) -> None:
        try:
            model_path = self._model_path_var.get().strip()
            aligner_path = self._aligner_path_var.get().strip()
            device = self._device_var.get()
            dtype_str = self._dtype_var.get()
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(dtype_str, torch.bfloat16)

            if not self._engine.is_loaded or self._engine.model_path != model_path:
                self._root.after(0, self._update_status, "正在加载模型...")
                self._engine.load_model(
                    model_path=model_path,
                    forced_aligner_path=aligner_path,
                    device=device,
                    dtype=dtype,
                )

            self._root.after(0, self._update_status, "正在识别...")

            language = self._language_var.get()
            if language == "自动检测":
                language = None

            return_timestamps = self._timestamps_var.get()
            if return_timestamps and not aligner_path:
                return_timestamps = False

            results = self._engine.transcribe(
                audio_path=self._audio_path,
                language=language,
                return_time_stamps=return_timestamps,
            )

            self._results = results
            self._root.after(0, self._on_transcribe_done)

        except Exception as e:
            self._root.after(0, self._on_transcribe_error, str(e))

    def _on_transcribe_done(self) -> None:
        self._set_ui_state(running=False)

        if not self._results:
            self._update_status("识别完成但无结果")
            messagebox.showwarning("提示", "未识别到任何文本")
            return

        r = self._results[0]
        fmt = self._output_format_var.get()
        custom_name = self._output_name_var.get().strip()

        # 确定输出路径: 与音频同目录, 文件名默认同音频名
        audio_dir = os.path.dirname(self._audio_path)
        if custom_name:
            base_name = custom_name
        else:
            base_name = os.path.splitext(os.path.basename(self._audio_path))[0]
        output_path = os.path.join(audio_dir, f"{base_name}.{fmt}")

        try:
            if r.time_stamps is not None:
                gap = float(self._gap_var.get())
                max_dur = float(self._max_dur_var.get())
                export_subtitle(r.text, r.time_stamps, fmt, output_path, gap, max_dur)
                self._update_status(f"已输出: {output_path}")
                _logger.info("已导出 %s 文件: %s", fmt.upper(), output_path)
            else:
                # 无时间戳时仅保存纯文本
                txt_path = os.path.join(audio_dir, f"{base_name}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(r.text)
                self._update_status(f"已输出纯文本: {txt_path}")
                _logger.info("已导出文本文件: %s", txt_path)
        except Exception as e:
            messagebox.showerror("输出失败", str(e))
            _logger.error("输出失败: %s", e)

    def _on_transcribe_error(self, error_msg: str) -> None:
        self._set_ui_state(running=False)
        self._update_status("识别失败")
        _logger.error("识别错误: %s", error_msg)
        messagebox.showerror("识别错误", error_msg)

    # UI 状态管理

    def _set_ui_state(self, running: bool) -> None:
        states = "disabled" if running else "normal"
        self._start_btn.configure(state=states)
        self._model_path_entry.configure(state=states)
        self._aligner_path_entry.configure(state=states)

        if running:
            self._progress.start()
        else:
            self._progress.stop()
            self._progress.set(0)

    def _update_status(self, msg: str) -> None:
        self._status_var.set(msg)

    # 启动

    def run(self) -> None:
        """启动主循环。"""
        self._root.mainloop()
