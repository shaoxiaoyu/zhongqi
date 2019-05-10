"""
https://www.cnblogs.com/pinard/p/7278324.html

训练得到 词向量模型
"""
import os
from tqdm import tqdm
from gensim.models import word2vec

from Config import Config
config = Config()


def get_text():
    f_all = open(config.all_file, "w")
    #  将 文本的主题 和 文本的内容 分开
    f_write = open(config.composition_text, "w")
    with open(config.composition) as f:
        line = f.readline().strip()
        while line:
            text, topic = line.split("</d>")
            f_write.write(text + "\n")
            f_all.write(text + "\n")
            line = f.readline().strip()
        f_write.close()

    f_write = open(config.zhihu_text, "w")
    with open(config.zhihu) as f:
        line = f.readline().strip()
        while line:
            text, topic = line.split("</d>")
            f_write.write(text + "\n")
            f_all.write(text + "\n")
            line = f.readline().strip()
        f_write.close()
    f_all.close()


if not os.path.exists(config.all_file):
    get_text()

if not os.path.exists(config.word2vec_model):
    sentences = word2vec.LineSentence(config.all_file)
    word2vec_model = word2vec.Word2Vec(sentences, hs=1, min_count=5, window=5, size=64)
    word2vec_model.save(config.word2vec_model)
else:
    # 加载model
    word2vec_model = word2vec.Word2Vec.load(config.word2vec_model)


vec_file = open(config.vec_file, "w")
vocab_keys = word2vec_model.wv.vocab.keys()
for w in tqdm(vocab_keys):
    line = [w]
    line.extend(map(str, list(word2vec_model[w])))
    line = " ".join(line) + "\n"
    vec_file.write(line)
vec_file.close()
