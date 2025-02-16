# ----------------------------------------
# 导入库，检测GPU
# ----------------------------------------

import tensorflow as tf

# 检查是否有可用的 GPU
print("TensorFlow version:", tf.__version__)
print(tf.test.is_built_with_cuda())  # 应该返回 True
print(tf.config.list_physical_devices('GPU'))  # 应该显示可用的 GPU

# ------------------------------
from PrepareData import PrepareData
from Transformer import Transformer 
from Test import test
# ------------------------------

# ----------------------------------------
# 超参数  
# ----------------------------------------

# 最大输入长度
MAX_LENGTH = 20

# 训练参数  
EPOCHS = 20 
batch_size = 128
#learning_rate = 0.0002

# 模型参数  
num_layers = 8
d_model = 256
dff = 512
num_heads = 8
dropout_rate = 0.1

# ----------------------------------------
# TensorBoard 日志目录
# ----------------------------------------
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/gradient_tape/' + current_time
train_log_dir = log_dir + '/train'
val_log_dir = log_dir + '/val'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)

# ----------------------------------------
# 训练
# ----------------------------------------

# 1.数据预处理
data_prep = PrepareData('data/spanish-to-english.csv', MAX_LENGTH=MAX_LENGTH, batch_size=batch_size)
train_dataset, val_dataset = data_prep.prepare_data()

# 2. 获取词汇表信息
input_vocab_size = data_prep.VOCAB_SIZE_SPANISH 
target_vocab_size = data_prep.VOCAB_SIZE_ENGLISH
START_TOKEN = data_prep.START_TOKEN 
END_TOKEN = data_prep.END_TOKEN
PAD_TOKEN = data_prep.PAD_TOKEN 
UNK_TOKEN = data_prep.UNK_TOKEN

# 3.模型构建
transformer = Transformer(
    num_layers,
    d_model,
    num_heads,
    dff,    
    input_vocab_size,
    target_vocab_size,
    MAX_LENGTH,
    rate=dropout_rate
)

# 4. 定义优化器，自定义学习率
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# 5. 定义损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, PAD_TOKEN))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / (tf.reduce_sum(mask) + 1e-8)

# 6. 定义监控指标
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

# 7. 添加 checkpoint 设置
checkpoint_path = "checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=transformer,
                          optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 8. 训练模型
@tf.function
def train_step(encoder_inputs, decoder_inputs, targets):
    with tf.GradientTape() as tape:
        predictions = transformer((encoder_inputs, decoder_inputs), training=True)
        loss = loss_function(targets, predictions)
        
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    
    # 更新指标
    train_loss(loss)
    train_accuracy(targets, predictions)
    
    return loss

def train():
    for epoch in range(EPOCHS):
        # 打印测试
        test(transformer, data_prep, train_dataset, 1)

        # 重置指标
        train_loss.reset_state()
        train_accuracy.reset_state()
        val_loss.reset_state()
        val_accuracy.reset_state()

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    
        # 打乱数据批次顺序
        dataset = train_dataset.shuffle(buffer_size=len(train_dataset))
        
        # 训练循环
        for step, (inputs, targets) in enumerate(dataset):
            encoder_inputs, decoder_inputs = inputs
            
            # 训练一个批次
            loss = train_step(encoder_inputs, decoder_inputs, targets)
            
            # 打印进度
            if step % 50 == 0:
                total_steps = len(train_dataset)
                print(f'Step {step}/{total_steps} - Epoch {epoch + 1}/{EPOCHS} - Loss: {train_loss.result():.4f}, Accuracy: {train_accuracy.result():.4f}')

                # 记录训练指标到 TensorBoard
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=epoch * len(train_dataset) + step)
                    tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch * len(train_dataset) + step)
                
            # 第1步打印模型summary
            if step == 0 and epoch == 0:
                transformer.summary()
        
        # 打印epoch结果
        print(f'\nEpoch {epoch + 1} - Loss: {train_loss.result():.4f}, Accuracy: {train_accuracy.result():.4f}')
        
        # 每2个epoch保存一次checkpoint
        if (epoch + 1) % 4 == 0:
            save_path = ckpt_manager.save()
            print(f'Saving checkpoint at epoch {epoch+1} at {save_path}')
        
        # 验证
        if val_dataset is not None:
            for val_inputs, val_targets in val_dataset:
                val_encoder_inputs, val_decoder_inputs = val_inputs
                predictions = transformer((val_encoder_inputs, val_decoder_inputs), training=False)
                v_loss = loss_function(val_targets, predictions)
                
                # 更新验证指标
                val_loss(v_loss)
                val_accuracy(val_targets, predictions)
            
            # 记录验证指标到 TensorBoard
            with val_summary_writer.as_default():
                tf.summary.scalar('loss', val_loss.result(), step=(epoch+1) * len(train_dataset))
                tf.summary.scalar('accuracy', val_accuracy.result(), step=(epoch+1) * len(train_dataset))
            
            print(f'Validation Loss: {val_loss.result():.4f}, Validation Accuracy: {val_accuracy.result():.4f}')
    
    # 9. 保存模型
    transformer.save_weights('models/transformer_final.h5')  

train()