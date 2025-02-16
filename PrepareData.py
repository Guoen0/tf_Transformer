import tensorflow as tf
import pandas as pd
import sys

class PrepareData:
    def __init__(self, path, MAX_LENGTH=40, batch_size=16):
        self.path = path
        self.data = None
        self.spanish_tokenizer = None
        self.english_tokenizer = None
        self.spanish_sequences = None
        self.english_sequences = None
        self.MAX_LENGTH = MAX_LENGTH
        self.batch_size = batch_size
        
        # 定义特殊token
        self.VOCAB_SIZE_SPANISH = 0  # 将在创建分词器后更新
        self.VOCAB_SIZE_ENGLISH = 0  # 将在创建分词器后更新
        self.PAD_TOKEN = 0   # 填充标记
        self.START_TOKEN = 1 # 句子开始标记
        self.END_TOKEN = 2   # 句子结束标记
        self.UNK_TOKEN = 3  # 未知词标记

        # 设置终端输出编码
        sys.stdout.reconfigure(encoding='utf-8')

    def load_data(self):
        """加载数据"""
        self.data = pd.read_csv(self.path, encoding='utf-8')
        return self.data

    def create_tokenizer(self, texts):
        """创建分词器"""
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='',  # 不过滤任何字符
            oov_token='<UNK>',  # 未知词标记
            lower=False  # 不转换为小写，保持原始大小写
        )
        
        # 先对文本进行拟合
        tokenizer.fit_on_texts(texts)
        
        # 保存原始的word_index
        original_word_index = tokenizer.word_index.copy()
        
        # 清空word_index
        tokenizer.word_index = {}
        
        # 按顺序添加特殊token
        tokenizer.word_index['<PAD>'] = self.PAD_TOKEN
        tokenizer.word_index['<START>'] = self.START_TOKEN
        tokenizer.word_index['<END>'] = self.END_TOKEN
        tokenizer.word_index['<UNK>'] = 3  # UNK token
        
        # 将原始词汇添加到word_index中，索引从4开始
        current_index = 4
        for word, _ in original_word_index.items():
            if word not in tokenizer.word_index:
                tokenizer.word_index[word] = current_index
                current_index += 1
        
        # 更新index_word
        tokenizer.index_word = {v: k for k, v in tokenizer.word_index.items()}
        
        return tokenizer

    def create_tokenizers(self):
        """创建源语言和目标语言的分词器"""
        self.spanish_tokenizer = self.create_tokenizer(self.data['Spanish'].values)
        self.english_tokenizer = self.create_tokenizer(self.data['English'].values)
        
        # 获取词汇表大小（包含特殊token）
        self.VOCAB_SIZE_SPANISH = len(self.spanish_tokenizer.word_index)
        self.VOCAB_SIZE_ENGLISH = len(self.english_tokenizer.word_index)
        
        print(f"西班牙语词汇表大小 (包含特殊token): {self.VOCAB_SIZE_SPANISH}")
        print(f"英语词汇表大小 (包含特殊token): {self.VOCAB_SIZE_ENGLISH}")
        
        # 打印特殊token的索引，用于验证
        print("\n特殊token索引:")
        print(f"PAD token: {self.spanish_tokenizer.word_index['<PAD>']}")
        print(f"START token: {self.spanish_tokenizer.word_index['<START>']}")
        print(f"END token: {self.spanish_tokenizer.word_index['<END>']}")
        print(f"UNK token: {self.spanish_tokenizer.word_index['<UNK>']}")
        
        return self.spanish_tokenizer, self.english_tokenizer
    
    # ------------------------------------------------------------

    def encode_and_pad(self, texts, tokenizer, sequence_type='encoder_input'):
        """编码和填充序列
        Args:
            texts: 要编码的文本列表
            tokenizer: 使用的分词器
            sequence_type: 序列类型，可选值：
                - 'encoder_input': 原始序列
                - 'decoder_input': START + 原始序列
                - 'target_output': 原始序列 + END
        """
        if sequence_type == 'encoder_input':
            # encoder输入：原始序列
            processed_texts = texts
        elif sequence_type == 'decoder_input':
            # decoder输入：START + 原始序列
            processed_texts = [f'<START> {text}' for text in texts]
        elif sequence_type == 'target_output':
            # 目标输出：原始序列 + END
            processed_texts = [f'{text} <END>' for text in texts]
        
        sequences = tokenizer.texts_to_sequences(processed_texts)
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequences,
            maxlen=self.MAX_LENGTH,
            padding='post',
            value=self.PAD_TOKEN
        )
        return padded

    def prepare_sequences(self):
        """准备编码后的序列"""
        # 准备源语言序列（西班牙语）
        self.spanish_sequences = self.encode_and_pad(
            self.data['Spanish'].values, 
            self.spanish_tokenizer,
            'encoder_input'
        )
        
        # 准备目标语言序列（英语）
        # decoder输入：START + 原始序列
        self.english_decoder_input = self.encode_and_pad(
            self.data['English'].values, 
            self.english_tokenizer,
            'decoder_input'
        )
        # 目标输出：原始序列 + END
        self.english_target = self.encode_and_pad(
            self.data['English'].values, 
            self.english_tokenizer,
            'target_output'
        )
        
        return self.spanish_sequences, self.english_decoder_input, self.english_target
    
    # ------------------------------------------------------------

    def create_dataset(self, src_sequences, decoder_inputs, target_sequences):
        """创建 TensorFlow 数据集
        返回格式：
            inputs: (encoder_inputs, decoder_inputs)
                - encoder_inputs: 原始西班牙语序列
                - decoder_inputs: START + 英语序列
            targets: 英语序列 + END
        """
        dataset = tf.data.Dataset.from_tensor_slices((
            # 模型输入：(encoder_inputs, decoder_inputs)
            {
                "encoder_inputs": src_sequences,
                "decoder_inputs": decoder_inputs
            },
            # 模型目标输出
            target_sequences
        ))
        
        dataset = dataset.shuffle(buffer_size=len(src_sequences))
        dataset = dataset.batch(self.batch_size)
        
        # 转换为模型需要的格式
        dataset = dataset.map(lambda x, y: (
            (x["encoder_inputs"], x["decoder_inputs"]),  # 模型输入
            y  # 模型目标输出
        ))
        
        return dataset

    def get_train_val_datasets(self):
        """获取训练集和验证集"""
        train_size = int(len(self.spanish_sequences) * 0.8)
        
        train_dataset = self.create_dataset(
            self.spanish_sequences[:train_size],
            self.english_decoder_input[:train_size],
            self.english_target[:train_size]
        )
        val_dataset = self.create_dataset(
            self.spanish_sequences[train_size:],
            self.english_decoder_input[train_size:],
            self.english_target[train_size:]
        )
        
        return train_dataset, val_dataset

    def prepare_data(self):
        """执行完整的数据准备流程"""
        self.load_data()
        self.create_tokenizers()
        self.prepare_sequences()

        # 获取训练集和验证集
        train_dataset, val_dataset = self.get_train_val_datasets()

        """
        # 打印1个批次的数据集信息，文本类型
        for spanish_batch, english_batch in train_dataset.take(1):
            print("\n批次示例:")
            print(f"Spanish batch shape: {spanish_batch.shape}")
            print(f"English batch shape: {english_batch.shape}")
            
            # 将数字序列转换回文本
            spanish_texts = []
            english_texts = []
            for spanish_seq, english_seq in zip(spanish_batch.numpy(), english_batch.numpy()):
                spanish_text = self.spanish_tokenizer.sequences_to_texts([spanish_seq])[0]
                english_text = self.english_tokenizer.sequences_to_texts([english_seq])[0]
                spanish_texts.append(spanish_text)
                english_texts.append(english_text)
                
            print("\nSpanish texts:", spanish_texts)
            print("English texts:", english_texts)
        """

        return train_dataset, val_dataset


# 调用方式
#data_prep = PrepareData('data/spanish-to-english.csv', batch_size=16)
#train_dataset, val_dataset = data_prep.prepare_data()
