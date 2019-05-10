class Config(object):

    # root = "/mnt/c/Users/v-lishao/Desktop/MTA-LSTM/"

    root = "./"
    data_dir = root + 'Data/'
    log = data_dir + "log.txt"
    # composition 和 zhihu 的数据的全部 doc 的文件
    all_file = data_dir + "composition_zhihu_text.txt"
    composition_text = data_dir + "composition_text.txt"
    composition = data_dir + "composition.txt"
    zhihu_text = data_dir + "zhihu_text.txt"
    zhihu = data_dir + "zhihu.txt"

    word2vec_model = data_dir + "word2vec.model"

    vec_file = data_dir + 'vec.txt'
    word_vec_path = data_dir + 'word_vec.pkl'
    word_voc_path = data_dir + 'word_voc.pkl'
    writer_path = data_dir + "coverage_data"  # epoch_size:15467  total step:  15466
    trainingData_path = data_dir + "composition.txt"
    trainingData_pickle = data_dir + "trainingdata.pickle"

    word_embedding_size = 64
    init_scale = 0.04
    learning_rate = 0.001
    max_grad_norm = 10  # gradient clipping
    num_layers = 2
    num_steps = 101
    # this value is one more than max number of words in sentence
    # 假设每句话最长不超过100个词，则此设置为101 多出来的一个 对应的是 START 或 END 的位置
    hidden_size = 32

    max_epoch = 10
    max_max_epoch = 20
    keep_prob = 0.8  # The probability that each element is kept through dropout layer
    lr_decay = 1.0
    batch_size = 32
    vocab_size = 493
    num_keywords = 5
    save_freq = 10  # The step (counted by the number of iterations) at which the model is saved to hard disk.
    model_path = './Model_News'  # the path of model that need to save or load
    
    # parameter for generation
    len_of_generation = 16  # The number of characters by generated
    save_time = 20  # load save_time saved models
    is_sample = True  # true means using sample, if not using argmax
    BeamSize = 2
