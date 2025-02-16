# Transformer 翻译任务测试

## 文件说明

- `Transformer.py`: 定义了 Transformer 模型。
- `PrepareData.py`: 定义了数据预处理函数。
- `Test.py`: 定义了测试函数。
- `data/spanish-to-english.csv`: 数据集，西班牙语到英语的翻译数据集。使用 `download_data.py` 下载。 


## 使用说明

- 运行 `main.py` 训练模型。训练完成后会自动保存模型参数到 `models/transformer_final.h5`
- 运行 `run_final_model_test.py` 自动加载预训练模型参数，测试模型效果