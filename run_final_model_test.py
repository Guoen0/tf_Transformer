import tensorflow as tf
from PrepareData import PrepareData
from Transformer import Transformer
from Test import test

# ----------------------------------------
# 测试预训练模型
# ----------------------------------------

MAX_LENGTH = 20
batch_size = 64
# 模型参数  
num_layers = 8
d_model = 256
dff = 512
num_heads = 8
dropout_rate = 0.1

# 1.数据预处理
data_prep = PrepareData('data/spanish-to-english.csv', MAX_LENGTH=MAX_LENGTH, batch_size=batch_size)
train_dataset, val_dataset = data_prep.prepare_data()

# 2.加载模型
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=data_prep.VOCAB_SIZE_SPANISH ,
    target_vocab_size=data_prep.VOCAB_SIZE_ENGLISH,
    maximum_position_encoding=MAX_LENGTH,
    rate=dropout_rate
)

# 创建一个虚拟输入来初始化模型变量
dummy_input = tf.ones((1, MAX_LENGTH), dtype=tf.int64)
dummy_output = tf.ones((1, MAX_LENGTH), dtype=tf.int64)
_ = transformer([dummy_input, dummy_output], training=False)

# 现在可以加载权重了
transformer.load_weights('models/transformer_final.h5')

# 3.测试
test(transformer, data_prep, val_dataset, 50)