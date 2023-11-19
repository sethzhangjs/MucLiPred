import argparse


def get_train_config():
    parse = argparse.ArgumentParser(description='ACPred-LAF train model')
    # cuda device
    parse.add_argument('-device', type=int, default=0, help='device id')

    # the loaded model path
    parse.add_argument('-load_model', type=str, default=None, help='load model path')

    # classificaiton complicate (orginal/simple 0, complicate 1, 2)
    parse.add_argument('-clsf_cmplct', type=int, default=0, help='complicate of classification')

    # *** 类别的个数 ***
    parse.add_argument('-class_num', type=int, default=2, help='number of class')

    # tmp
    # 当 class_num = 2: 没有用到tmp
    # 当 class_num = 3: 对于peptide，1 bert type token '0'；2 none; 3 manual type token '2';
    parse.add_argument('-tmp', type=int, default=0, help='tmp')

    # train_with (dna 0, rna 1)
    parse.add_argument('-train_with', type=int, default=0, help='peptide be trained with dna or rna')

    # **** Embedding Type ****
    # Embedding是否引入type信息
    parse.add_argument('-if_type', type=bool, default=True, help='if using type information')  # default True
    # Embedding引入type信息的方式
    # 0: Bert type token; 1: manual type embedding; 2: pretrained type embedding
    parse.add_argument('-type_method', type=int, default=0, help='type information method')  # default 0 bert type token

    # **** Classification Type ****
    # 向分类层引入type信息的方式
    # 0: none; 1: add; 2: concat
    parse.add_argument('-type_cls_method', type=int, default=1, help='cls type information method')  # default 1 add
    # block_3 的位置
    # 0: none; 1: inside; 2: outside
    parse.add_argument('-block_3_pos', type=int, default=1, help='block_3 position')  # default 1 inside

    # **** Contrastive Type ****
    # type对比学习的方式
    # 0: none; 1: encoder_output[0]; 2: pooler_output; 3: mean_pooling
    parse.add_argument('-type_contras_method', type=int, default=2, help='type contrastive method')  # default 2 pooler_output

    # **** 在BERT后是否使用block1（FNN）*** #
    # 1:use; 0: not use
    parse.add_argument('-if_block1', type=int, default=1, help='if using block1')  # default 1 use

    # **** Loss 占比 ****
    # CE loss proportion
    parse.add_argument('-ce_p', type=float, default=1, help='proportion of CE loss')
    # rsd loss proportion
    parse.add_argument('-rsd_p', type=float, default=1, help='proportion of residue contrastive learning')
    # type loss proportion
    parse.add_argument('-type_p', type=float, default=1, help='proportion of type contrastive learning')

    # CE loss weight
    parse.add_argument('-ce_weight', type=float, default=35, help='weight of CE loss')

    # *** 平滑梯度 ***
    # accum_times
    parse.add_argument('-accum_times', type=int, default=1, help='accumulation times')

    # *** 学习率 lr ***
    parse.add_argument('-lr', type=float, default=1e-5, help='learning rate')

    # *** test set ***
    parse.add_argument('-nt', type=str, default='rna', help='nucleic acid type')  
    parse.add_argument('-minsl', type=str, default='200', help='min length of input sequences')
    parse.add_argument('-maxsl', type=str, default='500', help='max length of input sequences')

    # *** train set ***
    parse.add_argument('-train_set_path', type=str, default='merge/merge_train_200_500.tsv', help='train set path')
    # parse.add_argument('-train_set_path', type=str, default='merge/equal_merge3_train_200_500.tsv', help='train set path')
    # parse.add_argument('-train_set_path', type=str, default='merge/merge2_dna_peptide_train_200_500.tsv', help='train set path')

    # Epoch
    parse.add_argument('-epoch', type=int, default=5, help='number of iteration')  # 10


    # preoject setting
    parse.add_argument('-learn-name', type=str, default='ACPred-LAF_train_00', help='learn name')

    parse.add_argument('-save-best', type=bool, default=True, help='if save parameters of the current best model ')
    # parse.add_argument('-save-best', type=bool, default=False, help='if save parameters of the current best model ')
    # save exp model
    parse.add_argument('-save_exp', type=str, default='exp4', help='save the model for certain experiment')


    parse.add_argument('-threshold', type=float, default=0.80, help='save threshold')

    # model parameters
    parse.add_argument('-max-len', type=int, default=256, help='max length of input sequences')
    parse.add_argument('-num-layer', type=int, default=2, help='number of encoder blocks')
    parse.add_argument('-num-head', type=int, default=8, help='number of head in multi-head attention')
    # parse.add_argument('-dim-embedding', type=int, default=128, help='residue embedding dimension')
    parse.add_argument('-dim-embedding', type=int, default=64, help='residue embedding dimension')
    # parse.add_argument('-dim-feedforward', type=int, default=128, help='hidden layer dimension in feedforward layer')
    parse.add_argument('-dim-feedforward', type=int, default=64, help='hidden layer dimension in feedforward layer')
    # parse.add_argument('-dim-k', type=int, default=64, help='embedding dimension of vector k or q')
    parse.add_argument('-dim-k', type=int, default=32, help='embedding dimension of vector k or q')
    # parse.add_argument('-dim-v', type=int, default=64, help='embedding dimension of vector v')
    parse.add_argument('-dim-v', type=int, default=32, help='embedding dimension of vector v')
    parse.add_argument('-num-embedding', type=int, default=2, help='number of sense in multi-sense')
    parse.add_argument('-k-mer', type=int, default=3, help='number of k(-mer) in multi-scaled')
    parse.add_argument('-embed-atten-size', type=int, default=8, help='size of soft attetnion')

    # parse.add_argument('-max-len', type=int, default=256, help='max length of input sequences')
    # parse.add_argument('-num-layer', type=int, default=3, help='number of encoder blocks')
    # parse.add_argument('-num-head', type=int, default=8, help='number of head in multi-head attention')
    # parse.add_argument('-dim-embedding', type=int, default=32, help='residue embedding dimension')
    # parse.add_argument('-dim-feedforward', type=int, default=32, help='hidden layer dimension in feedforward layer')
    # parse.add_argument('-dim-k', type=int, default=32, help='embedding dimension of vector k or q')
    # parse.add_argument('-dim-v', type=int, default=32, help='embedding dimension of vector v')
    # parse.add_argument('-num-embedding', type=int, default=2, help='number of sense in multi-sense')
    # parse.add_argument('-k-mer', type=int, default=3, help='number of k(-mer) in multi-scaled')
    # parse.add_argument('-embed-atten-size', type=int, default=8, help='size of soft attetnion')

    # training parameters
    parse.add_argument('-model_name', type=str, default='ACPred_LAF_Basic', help='model name')
    parse.add_argument('-if_multi_scaled', type=bool, default=False, help='if using k-mer ')
    # parse.add_argument('-lr', type=float, default=1e-4, help='learning rate') # 1e-5
    # parse.add_argument('-lr', type=float, default=0.0005, help='learning rate')
    # parse.add_argument('-reg', type=float, default=0.0025, help='weight lambda of regularization')
    parse.add_argument('-reg', type=float, default=1e-5, help='weight lambda of regularization')
    # parse.add_argument('-batch-size', type=int, default=64, help='number of samples in a batch')


    # batch size
    parse.add_argument('-batch-size', type=int, default=2, help='number of samples in a batch')
    # batch size type
    parse.add_argument('-batch-size-type', type=str, default=5, help='batch size type')

    # parse.add_argument('-batch-size', type=int, default=16, help='number of samples in a batch')

    #
    parse.add_argument('-k-fold', type=int, default=-1, help='k in cross validation,-1 represents train-test approach')
    # parse.add_argument('-k-fold', type=int, default=5, help='k in cross validation,-1 represents train-test approach')
    parse.add_argument('-num-class', type=int, default=2, help='number of classes')
    parse.add_argument('-cuda', type=bool, default=True, help='if use cuda')

    parse.add_argument('-interval-log', type=int, default=100,
                       help='how many batches have gone through to record the training performance')
    parse.add_argument('-interval-valid', type=int, default=1,
                       help='how many epoches have gone through to record the validation performance')
    parse.add_argument('-interval-test', type=int, default=1,
                       help='how many epoches have gone through to record the test performance')

    config = parse.parse_args()
    return config
