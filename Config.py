class Config(object):

    root = "./"
    data_dir = root + 'Data/'
    log = data_dir + "log.txt"

    #######################################################
    composition = data_dir + "composition.txt"
    composition_doc = data_dir + "composition_doc.txt"

    word_embedding_size = 64
    mode = "_only_composition_dim64"
    word2vec_model = data_dir + "word2vec" + mode + ".model"
    vec_file = data_dir + "vec" + mode + ".txt"
    #######################################################

    word_vec_path = data_dir + "word_vec" + mode + ".pickle"
    word_voc_path = data_dir + "word_voc" + mode + ".pickle"

    trainData_pickle = data_dir + "trainData" + mode + ".pickle"  # list 存放 (doc, keywords)
    writer_path = data_dir + "coverage_data" + mode  # epoch_size:15467  total step:  15466

    #######################################################
    trainingData_path = data_dir + "composition.txt"



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
    vocab_size = 82461  # Preprocess.py 得到的
    num_keywords = 5
    save_freq = 1  # The step (counted by the number of iterations) at which the model is saved to hard disk.
    model_path = './Model_News'  # the path of model that need to save or load
    
    # parameter for generation
    len_of_generation = 16  # The number of characters by generated
    save_time = 20  # load save_time saved models
    is_sample = True  # true means using sample, if not using argmax
    BeamSize = 2
