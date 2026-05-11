# Qwen3ASRer
此项目仍处于早期测试阶段，不保证可用性，不保证稳定性。

## 技术信息
### 开发环境
- Python 3.12.10
- PyTorch CU129 (CUDA Toolkit 12.9)
- 详见`requirements.txt`
### 程序设计
- 使用Transformers后端运行Qwen3-ASR，支持使用Qwen3ForcedAligner对齐时间戳。
- 附有基本的GUI界面，方便快速测试。
- 模块化架构，为后续整合进其他项目做铺垫，因此该项目可能会被迁移或存档。