import numpy as np
import tensorflow as tf
import pickle
import os
from tqdm import tqdm
from Config import Config
config = Config()


def save_pickle(content, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj=content, file=handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as handle:
        content = pickle.load(handle)
    return content


def Read_WordVec():
    if not os.path.exists(config.word_vec_path):
        with open(config.vec_file, 'r', encoding="utf-8") as fvec:
            word_voc = []
            vec_ls = []

            word_voc.append('PAD')
            vec_ls.append([0]*config.word_embedding_size)
            word_voc.append('START')
            vec_ls.append([0]*config.word_embedding_size)
            word_voc.append('END')
            vec_ls.append([0]*config.word_embedding_size)
            word_voc.append('UNK')
            vec_ls.append([0]*config.word_embedding_size)

            for line in fvec:
                line = line.split()
                try:
                    word = line[0]
                    vec = [float(i) for i in line[1:]]
                    assert len(vec) == config.word_embedding_size
                    word_voc.append(word)
                    vec_ls.append(vec)
                except:
                    print(line[0])

            config.vocab_size = len(word_voc)
            word_vec = np.array(vec_ls, dtype=np.float32)

            save_pickle(word_vec, config.word_vec_path)
            save_pickle(word_voc, config.word_voc_path)
    else:
        word_vec = load_pickle(config.word_vec_path)
        word_voc = load_pickle(config.word_voc_path)

    return word_voc, word_vec


def Read_Data():
    train = []
    if not os.path.exists(config.trainData_pickle):
        with open(config.trainingData_path, 'r', encoding="utf-8") as ftext:
            for line in ftext:
                tmp = line.split()
                idx = tmp.index('</d>')
                doc = tmp[:idx]
                keywords = tmp[idx+1:]
                assert len(keywords) == 5
                train.append((doc, keywords))
            save_pickle(train, config.trainData_pickle)
    else:
        train = load_pickle(config.trainData_pickle)

    # 统计一下 语料中 doc 的平均长度  composition 是 65
    len_doc = []
    for doc, topic in train:
        len_doc.append(len(doc))
    ave_len  = sum(len_doc)/len(len_doc)
    print("语料中 doc 的平均长度:{}".format(ave_len))
    return train


print('loading the training data...')
vocab, _ = Read_WordVec()

data = Read_Data()

word_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_word = {i: ch for i, ch in enumerate(vocab)}
data_size, _vocab_size = len(data), len(vocab)
print('data has %d document, size of word vocabulary: %d.' % (data_size, _vocab_size))


def data_iterator(train_data, batch_size, num_steps):
    epoch_size = len(train_data) // batch_size
    print("epoch_size:{}".format(epoch_size))
    for i in range(epoch_size):
        batch_data = train_data[i * batch_size : (i+1) * batch_size]
        data_x = np.zeros((batch_size, num_steps), dtype=np.int64)
        data_y = np.zeros((batch_size, num_steps), dtype=np.int64)
        key_words = []

        ids = 0
        for it in batch_data:
            doc = it[0]
            topic = it[1]

            key_words.append([word_to_idx.get(wd, 3) for wd in topic])

            doc = [word_to_idx.get(wd, 3) for wd in doc]
            doc = doc[:num_steps-1]
            # num_steps 是 max len(doc) + 1
            # doc 的长度超过100的话 只取前100

            doc_x = [1] + doc
            doc_x = np.array(doc_x, dtype=np.int64)
            data_x[ids][:len(doc_x)] = doc_x

            doc_y = doc + [2]
            doc_y = np.array(doc_y, dtype=np.int64)
            data_y[ids][:len(doc_y)] = doc_y

            ids += 1

        key_words = np.array(key_words, dtype=np.int64)
        mask = np.float32(data_x != 0)
        yield (data_x, data_y, mask, key_words)
            
if not os.path.exists(config.writer_path):
    train_data = data
    writer = tf.python_io.TFRecordWriter(config.writer_path)
    iterator = data_iterator(train_data, config.batch_size, config.num_steps)

    step = 0
    for x, y, mask, key_words in tqdm(iterator):
        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(
              # Features contains a map of string to Feature proto objects
              feature={
                # A Feature contains one of either a int64_list, float_list, or bytes_list
                'input_data': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=x.reshape(-1).astype("int64"))),
                'target': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=y.reshape(-1).astype("int64"))),
                'mask': tf.train.Feature(
                    float_list=tf.train.FloatList(value=mask.reshape(-1).astype("float"))),
                'key_words': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=key_words.reshape(-1).astype("int64")))
              }))
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)
        step += 1

    print('total step: ', step)


"""
loading the training data...
data has 494944 document, size of word vocabulary: 82461.
0it [00:00, ?it/s]epoch_size:15467
15467it [00:58, 265.64it/s]
total step:  15467
"""
