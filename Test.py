import tensorflow as tf

def test(transformer, data_prep, dataset, num_samples=1):

    # 从 val_dataset 中随机选择 1 批次
    val_sample = next(iter(dataset.take(1)))
    val_inputs, val_targets = val_sample
    val_encoder_inputs, val_decoder_inputs = val_inputs

    for i in range(num_samples):

        # 选择1个样本并扩展维度使其成为batch   
        input = tf.expand_dims(val_encoder_inputs[i], 0)  # 添加batch维度
        target = tf.expand_dims(val_targets[i], 0)  # 添加batch维度

        # 将输入token转换为西班牙语文字
        input_text = data_prep.spanish_tokenizer.sequences_to_texts(input.numpy())
        # 将目标token转换为英语文字
        target_text = data_prep.english_tokenizer.sequences_to_texts(target.numpy())
        
        print(f'输入(西班牙语): {input_text}')
        print(f'目标翻译(英语): {target_text}')

        # 获取预测结果
        predicted_tokens = predict(transformer, input, data_prep.START_TOKEN, data_prep.END_TOKEN, data_prep.MAX_LENGTH)
        # 将预测的token转换为英语文字
        predicted_text = data_prep.english_tokenizer.sequences_to_texts(predicted_tokens.numpy())
        print("--------------------------------")
        print(f'模型翻译(英语): {predicted_text}')
        print("--------------------------------")

def predict(transformer, encoder_inputs, START_TOKEN, END_TOKEN, MAX_LENGTH):
    
    # 创建解码器输入（以START_TOKEN开始）
    batch_size = tf.shape(encoder_inputs)[0]
    decoder_inputs = tf.ones((batch_size, 1), dtype=tf.int64) * START_TOKEN
    
    output = tf.zeros((batch_size, 0), dtype=tf.int64)
    
    for i in range(MAX_LENGTH):
        predictions = transformer([encoder_inputs, decoder_inputs], training=False)
        
        # 获取最后一个时间步的预测
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.argmax(predictions, axis=-1)  # (batch_size, 1)
        
        # 将预测添加到输出
        output = tf.concat([output, predicted_id], axis=-1)
        
        # 如果预测到了END_TOKEN，就停止预测
        if predicted_id == END_TOKEN:
            break
            
        # 更新decoder_inputs
        decoder_inputs = tf.concat([decoder_inputs, predicted_id], axis=-1)
    
    return output