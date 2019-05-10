"""
https://www.cnblogs.com/pinard/p/7278324.html

训练得到 词向量模型 并且 把训练得到的词向量写到txt文件中
"""
import os
import time
from tqdm import tqdm
from gensim.models import word2vec

from Config import Config
config = Config()


def get_doc():
    #  将 文本的主题 和 文本的内容 分开
    f_write = open(config.composition_doc, "w")
    with open(config.composition) as f:
        line = f.readline().strip()
        while line:
            doc, topic = line.split("</d>")
            f_write.write(doc + "\n")
            line = f.readline().strip()
        f_write.close()


if not os.path.exists(config.composition_doc):
    get_doc()


if not os.path.exists(config.word2vec_model):
    print("开始训练词向量")
    start = time.time()
    sentences = word2vec.LineSentence(config.composition_doc)

    word2vec_model = word2vec.Word2Vec(sentences,
                                       hs=1, min_count=5, window=5,
                                       size=config.word_embedding_size)

    word2vec_model.save(config.word2vec_model)
    print("保存成功，词向量训练用时：{}".format(time.time() - start))
else:
    print("开始加载词向量")
    word2vec_model = word2vec.Word2Vec.load(config.word2vec_model)

print("词向量写入到txt中")
vec_file = open(config.vec_file, "w")
vocab_keys = word2vec_model.wv.vocab.keys()
for w in tqdm(vocab_keys):
    line = [w]
    line.extend(map(str, list(word2vec_model[w])))
    line = " ".join(line) + "\n"
    vec_file.write(line)
vec_file.close()
