import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


from configuration import config as cf
from util import util_metric
from train.model_operation import save_model, adjust_model
from train.visualization import dimension_reduction, penultimate_feature_visulization
from model import prot_bert
from util import data_loader_protBert

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import pickle
import seaborn as sns
import random

cfg = cf.get_train_config()

data_root_path = '/home/sde3/wrh/zjs/PepBCL/data/'

# 例：使用最小长度为20，最大长度为500，类型为dna的数据集进行训练
# python protBert_main.py -nt dna -minsl 20 -maxsl 500
# nohup python protBert_main.py -nt dna -minsl 20 -maxsl 500 > ../../log/dna/dna_20_500_2_10.txt 2>&1 &
# nohup bash cmd.sh 2>&1 &

nucleic_type = cfg.nt
min_seq_len = cfg.minsl
max_seq_len = cfg.maxsl

# type对比学习的方式 0: none; 1: encoder_output[0]; 2: pooler_output; 3: mean_pooling
type_contras_method = cfg.type_contras_method

train_set_path = cfg.train_set_path
train_path = data_root_path + train_set_path
test_path = data_root_path + 'rna/rna_predict_dna_200_500.tsv'

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# negative cosine similarity
def D(p, z, version='simplified'):
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return 1 - F.cosine_similarity(p, z, dim=-1)
    else:
        raise Exception

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        # euclidean_distance = F.pairwise_distance(output1, output2)
        cos_distance = D(output1, output2)
        # print("ED",euclidean_distance)
        loss_contrastive = torch.mean((1 - label) * torch.pow(cos_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - cos_distance, min=0.0), 3))

        return loss_contrastive

class ContrastiveTypeLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveTypeLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cos_distance = D(output1, output2)
        # label = 1 if different type, 0 if same type
        loss_type_contrastive = torch.mean((1 - label) * torch.pow(cos_distance, 2) + label * torch.pow(torch.clamp(self.margin - cos_distance, min=0.0), 2))

        return loss_type_contrastive

def get_xor_result(x, y):
    if x == y:
        # same type 0
        return 0
    else:
        # different type 1
        return 1

def load_data(config):
    train_iter_orgin, test_iter = data_loader_protBert.load_data(config)
    # print('-' * 20, 'data construction over', '-' * 20)
    return train_iter_orgin, test_iter


def cal_loss_dist_by_cosine(model):
    embedding = model.embedding
    loss_dist = 0

    vocab_size = embedding[0].tok_embed.weight.shape[0]
    d_model = embedding[0].tok_embed.weight.shape[1]

    Z_norm = vocab_size * (len(embedding) ** 2 - len(embedding)) / 2

    for i in range(len(embedding)):
        for j in range(len(embedding)):
            if i < j:
                cosin_similarity = torch.cosine_similarity(embedding[i].tok_embed.weight, embedding[j].tok_embed.weight)
                loss_dist -= torch.sum(cosin_similarity)
                # print('cosin_similarity.shape', cosin_similarity.shape)
    loss_dist = loss_dist / Z_norm
    return loss_dist


def get_loss(logits, label, criterion):
    loss = criterion(logits, label)
    loss = loss.float()
    # flooding method
    loss = (loss - config.b).abs() + config.b
    return loss

def get_val_loss(logits, label, criterion):
    loss = criterion(logits.view(-1, config.num_class), label.view(-1))
    loss = (loss.float()).mean()
    # flooding method
    loss = (loss - config.b).abs() + config.b
    Q_sum = len(logits)
    logits = F.softmax(logits, dim=1)  # softmax归一化
    hat_sum_p0 = logits[:, 0].sum()/Q_sum  # 负类的概率和
    hat_sum_p1 = logits[:, 1].sum()/Q_sum  # 正类的概率和
    mul_hat_p0 = hat_sum_p0.mul(torch.log(hat_sum_p0))
    mul_hat_p1 = hat_sum_p1.mul(torch.log(hat_sum_p1))
    mul_p0 = logits[:, 0].mul(torch.log(logits[:, 0])).sum()/Q_sum
    mul_p1 = logits[:, 1].mul(torch.log(logits[:, 1])).sum()/Q_sum
    # sum_loss = loss+(-1)*(mul_hat_p0+mul_hat_p1) + 0.1*(mul_p0+mul_p1)
    sum_loss = loss+(mul_hat_p0+mul_hat_p1)-0.1*(mul_p0+mul_p1)
    return sum_loss

def periodic_test(test_iter, model, criterion, config, sum_epoch):
    print('#' * 60 + 'Periodic Test' + '#' * 60)
    test_metric, test_loss, test_repres_list, test_label_list, \
    test_roc_data, test_prc_data = model_eval(test_iter, model, criterion, config)

    print('test current performance')
    plmt = test_metric.numpy()
    print('[ACC,\t\tPrecision,\tSensitivity,\tSpecificity,\tF1,\t\t\tAUC,\t\tMCC,\t\tTP,\t\t\tFP,\t\t\tTN,\t\t\tFN\t]')
    print('{:.5f}'.format(plmt[0]), '\t{:.5f}'.format(plmt[1]), '\t{:.5f}'.format(plmt[2]),'\t\t{:.5f}'.format(plmt[3]),
          '\t\t{:.5f}'.format(plmt[4]), '\t{:.5f}'.format(plmt[5]), '\t{:.5f}'.format(plmt[6]),'\t{:.0f}'.format(plmt[7]),
          '\t\t{:.0f}'.format(plmt[8]), '\t\t{:.0f}'.format(plmt[9]), '\t\t{:.0f}'.format(plmt[10]))

    print('#' * 60 + 'Over' + '#' * 60)

    step_test_interval.append(sum_epoch)
    test_acc_record.append(test_metric[0])
    test_loss_record.append(test_loss)

    return test_metric, test_loss, test_repres_list, test_label_list


def periodic_valid(valid_iter, model, criterion, config, sum_epoch):
    print('#' * 60 + 'Periodic Validation' + '#' * 60)

    valid_metric, valid_loss, valid_repres_list, valid_label_list, \
    valid_roc_data, valid_prc_data = model_eval(valid_iter, model, criterion, config)

    print('validation current performance')
    print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
    print(valid_metric.numpy())
    print('#' * 60 + 'Over' + '#' * 60)

    step_valid_interval.append(sum_epoch)
    valid_acc_record.append(valid_metric[0])
    valid_loss_record.append(valid_loss)

    return valid_metric, valid_loss, valid_repres_list, valid_label_list


def train_model(train_iter, valid_iter, test_iter, model, optimizer, criterion, contras_criterion, contras_type_criterion , config, iter_k):
    best_acc = 0
    best_performance = 0
    train_batch_loss = 0
    for epoch in range(1, config.epoch + 1):
        steps = 0
        train_epoch_loss = 0
        train_correct_num = 0
        train_total_num = 0
        current_batch_size = 0
        repres_list = []
        label_list = []

        label_b = []
        output_b = []
        logits_b = []

        # *** modification ***
        CLS_b = []
        type_b = []
        # *** modification ***

        model.train()
        random.shuffle(train_iter)
        for batch in train_iter:
            input, label, type = batch
            label = torch.tensor(label, dtype=torch.long).cuda()
            type = torch.tensor(type, dtype=torch.long).cuda()
            output, pooler_output = model.forward(input, type)
            logits = model.get_logits(input, type)

            output = output.view(-1, output.size(-1))
            logits = logits.view(-1, logits.size(-1))

            if type_contras_method == 1:
                CLS_b.append(output[0])
            elif type_contras_method == 2:
                CLS_b.append(pooler_output)

            # remove [CLS] and [SEP]
            label = label[1:-1]
            logits = logits[1:-1]
            output = output[1:-1]

            # use mean pooling instead of CLS
            if type_contras_method == 3:
                CLS_b.append(torch.mean(output, dim=0))

            # add to the list
            output_b.append(output)
            logits_b.append(logits)
            label_b.append(label)
            type_b.append(type)

            # *** Start Calculating Loss ***
            current_batch_size += 1
            if current_batch_size % config.batch_size == 0:

                # input 1
                output_1 = output_b[0]
                label_1 = label_b[0]

                # input 2
                output_2 = output_b[1]
                label_2 = label_b[1]

                label_1 = label_1.view(-1)
                output_1 = output_1.view(-1, output_1.size(-1))

                label_2 = label_2.view(-1)
                output_2 = output_2.view(-1, output_2.size(-1))

                # used for CE loss
                label_b = torch.cat(label_b, dim=0)
                label_b = label_b.view(-1)

                logits_b = torch.cat(logits_b, dim=0)
                logits_b = logits_b.view(-1, logits_b.size(-1))

                # input 1
                label_ls_1 = []
                contras_len_1 = len(output_1) // 2
                label1_1 = label_1[:contras_len_1]
                label2_1 = label_1[contras_len_1:contras_len_1 * 2]
                for i in range(contras_len_1):
                    xor_label = (label1_1[i] ^ label2_1[i])
                    label_ls_1.append(xor_label.unsqueeze(0))

                contras_label_1 = torch.cat(label_ls_1)
                output1_1 = output_1[:contras_len_1]
                output2_1 = output_1[contras_len_1:contras_len_1 * 2]
                contras_loss_1 = contras_criterion(output1_1, output2_1, contras_label_1)

                # input 2
                label_ls_2 = []
                contras_len_2 = len(output_2) // 2
                label1_2 = label_2[:contras_len_2]
                label2_2 = label_2[contras_len_2:contras_len_2 * 2]
                for i in range(contras_len_2):
                    xor_label = (label1_2[i] ^ label2_2[i])
                    label_ls_2.append(xor_label.unsqueeze(0))

                contras_label_2 = torch.cat(label_ls_2)
                output1_2 = output_2[:contras_len_2]
                output2_2 = output_2[contras_len_2:contras_len_2 * 2]
                contras_loss_2 = contras_criterion(output1_2, output2_2, contras_label_2)

                # type contras loss
                if type_contras_method != 0:
                    contras_loss_tp = contras_type_criterion(CLS_b[0], CLS_b[1], get_xor_result(type_b[0], type_b[1]))

                # cross entropy loss
                ce_loss = criterion(logits_b, label_b)

                # FINAL loss
                ce_w = config.ce_p
                tp_w = config.type_p
                rsd_w = config.rsd_p
                if type_contras_method == 0:
                    loss = ce_w * ce_loss + rsd_w * (contras_loss_1 + contras_loss_2)
                else:
                    # print("type_contras_loss added")
                    loss = ce_w * ce_loss + rsd_w * (contras_loss_1 + contras_loss_2) + tp_w * contras_loss_tp


                train_batch_loss = loss.item()
                train_epoch_loss += train_batch_loss

                loss = loss / config.accum_times
                loss.backward()

                if current_batch_size % (config.accum_times * config.batch_size) == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                corre = (torch.max(logits_b, 1)[1] == label_b).int()
                corrects = corre.sum()
                train_correct_num += corrects
                the_batch_size = label_b.size(0)
                train_total_num += the_batch_size
                train_acc = 100.0 * corrects / the_batch_size

                label_b = []
                output_b = []
                logits_b = []

                CLS_b = []
                type_b = []

                steps = steps + 1

                '''Periodic Train Log'''
                if steps != 0 and steps % config.interval_log == 0:
                    sys.stdout.write(
                        '\rEpoch[{}] Batch[{}] - loss: {:.6f} | ACC: {:.4f}%({}/{})'.format(epoch, steps,
                                                                                            train_batch_loss,
                                                                                            train_acc,
                                                                                            corrects,
                                                                                            the_batch_size))
                    print()

                    step_log_interval.append(steps)
                    train_acc_record.append(train_acc)
                    train_loss_record.append(train_batch_loss)



        sum_epoch = iter_k * config.epoch + epoch
        print(f"Train - Epoch[{epoch}] - loss: {train_epoch_loss/(len(train_iter)//config.batch_size)} | ACC: {(train_correct_num/train_total_num)*100:.4f}%({train_correct_num}/{train_total_num})")

        '''Periodic Validation'''
        if valid_iter and sum_epoch % config.interval_valid == 0:
            valid_metric, valid_loss, valid_repres_list, valid_label_list = periodic_valid(valid_iter,
                                                                                           model,
                                                                                           criterion,
                                                                                           config,
                                                                                           sum_epoch)
            valid_acc = valid_metric[0]
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_performance = valid_metric

        '''Periodic Test'''
        if test_iter and sum_epoch % config.interval_test == 0:
            time_test_start = time.time()

            test_metric, test_loss, test_repres_list, test_label_list = periodic_test(test_iter,
                                                                                      model,
                                                                                      criterion,
                                                                                      config,
                                                                                      sum_epoch)
            '''Periodic Save'''
            # save the model if specific conditions are met
            test_F1 = test_metric[4]
            test_acc = test_metric[5] # auc actually
            test_mcc = test_metric[6]
            if test_acc > best_acc:
                best_acc = test_acc
                best_performance = test_metric

                # (case) Save the model
                if config.save_best and best_acc > 0.89 and test_mcc > 0.44 and test_F1 > 0.46:
                    # save the complete model
                    torch.save(model, f'Case_AUC={best_acc:.03f}; MCC={test_mcc:.03f}; F1={test_F1:.03f}.pl')

            # test_label_list = [x + 2 for x in test_label_list]
            repres_list.extend(test_repres_list)
            label_list.extend(test_label_list)

            '''feature dimension reduction'''
            # if sum_epoch % 1 == 0 or epoch == 1:
            #      dimension_reduction(repres_list, label_list, epoch)

            '''reduction feature visualization'''
            # if sum_epoch % 5 == 0 or epoch == 1 or (epoch % 2 == 0 and epoch <= 10):
            #     penultimate_feature_visulization(repres_list, label_list, epoch)
            #
            # time_test_end = time.time()
            # print('inference time:', time_test_end - time_test_start, 'seconds')

    return best_performance


def model_eval(data_iter, model, criterion, config):
    device = torch.device("cuda" if config.cuda else "cpu")
    label_pred = torch.empty([0], device=device)
    label_real = torch.empty([0], device=device)
    pred_prob = torch.empty([0], device=device)

    print('model_eval data_iter', len(data_iter))

    iter_size, corrects, avg_loss = 0, 0, 0
    repres_list = []
    label_list = []

    model.eval()

    with torch.no_grad():
        # random.shuffle(data_iter)
        for batch in data_iter:
            # input, label = batch
            # *** modification ***
            input, label, type = batch
            # input = input.cuda()
            label = torch.tensor(label, dtype=torch.long).cuda()
            # *** modification ***
            type = torch.tensor(type, dtype=torch.long).cuda()
            # pssm = torch.tensor(pssm, dtype=torch.float).cuda()
            # input = torch.unsqueeze(input, 0)
            lll = label.clone()
            label = torch.unsqueeze(label, 0)
            # pssm = torch.unsqueeze(pssm, 0)
            # 修改
            # label = label.view(-1)
            # logits = model.get_logits(input)
            logits = model.get_logits(input, type)
            # output = model.forward(input)
            # *** modification ***
            output, pooler_output = model.forward(input, type)
            # logits = torch.unsqueeze(logits[:, :2], 0)

            repres_list.extend(output.cpu().detach().numpy())
            label_list.extend(lll.cpu().detach().numpy())

            # loss = criterion(logits.view(-1, config.num_class), label.view(-1))
            logits = logits.view(-1, logits.size(-1))
            label = label.view(-1)
            label = label[1:-1]
            logits = logits[1:-1]
            loss = criterion(logits, label)
            # loss = (loss.float()).mean()
            avg_loss += loss.item()

            logits = torch.unsqueeze(logits, 0)
            label = torch.unsqueeze(label, 0)
            pred_prob_all = F.softmax(logits, dim=2)
            # Prediction probability [batch_size, seq_len, class_num]
            pred_prob_positive = pred_prob_all[:, :, 1]
            positive = torch.empty([0], device=device)
            # Probability of predicting positive classes [batch_size, seq_len]
            pred_prob_sort = torch.max(pred_prob_all, 2)
            # The maximum probability of prediction in each sample [batch_size]
            pred_class = pred_prob_sort[1]
            p_class = torch.empty([0], device=device)
            la = torch.empty([0], device=device)

            print("Seq: ")
            print(input)
            print("Type: ")
            cur_type = ''
            if(type.item() == 0):
                cur_type = "0 - DNA"
            elif(type.item() == 1):
                cur_type = "1 - RNA"
            elif(type.item() == 2):
                cur_type = "2 - Peptide"
            print(cur_type)
            print("Label: ")
            # print(label.view(-1).cpu().numpy().tostring())
            label_tmp = label.view(-1)
            label_string = ''.join([str(i) for i in label_tmp.tolist()])
            print(label_string)
            print("Prediction: ")
            # print(pred_class.view(-1).cpu().numpy().tostring())
            predict_tmp = pred_class.view(-1)
            predict_string = ''.join([str(i) for i in predict_tmp.tolist()])
            print(predict_string)
            print('*=' * 100)

            # 我们的目的是预测一个未知label的蛋白质序列的label，所以不需要进行评价，只需要结果。
            # 于是，把后面计算指标的代码都注释了。

            # The location (class) of the predicted maximum probability in each sample [batch_size, seq_len]
            # batch_pro_len = 0
            # m = torch.zeros_like(label)
            # B, seq_len = label.size()
            # for i in range(B):
            # pro_len = label.size(1)
            #     batch_pro_len += pro_len
            # positive = torch.cat([positive, pred_prob_positive[0][:]])
            # p_class = torch.cat([p_class, pred_class[0][:]])
            # la = torch.cat([la, label[0][:]])
            #     # for j in range(1, pro_len + 1):
            #     #     m[i][j] = 1
            #
            # corre = (pred_class == label).int()
            # # corre = torch.mm(corre, m.t())
            # # index = torch.arange(0, B).view(1, -1)
            # # corre = corre.gather(0, index)
            # # corre = torch.mul(corre, m)
            # corrects += corre.sum()
            # iter_size += label.size(1)
            # label_pred = torch.cat([label_pred, p_class.float()])
            # label_real = torch.cat([label_real, la.float()])
            # pred_prob = torch.cat([pred_prob, positive])


    # metric, roc_data, prc_data = util_metric.caculate_metric(label_pred, label_real, pred_prob)
    # avg_loss /= len(data_iter)
    # # accuracy = 100.0 * corrects / iter_size
    # accuracy = metric[0]
    # print('Evaluation - loss: {:.6f}  ACC: {:.4f}%({}/{})'.format(avg_loss,
    #                                                               100*accuracy,
    #                                                               corrects,
    #                                                             iter_size))
    metric = [0, 0, 0, 0]
    avg_loss = 0
    repres_list = []
    label_list = []
    roc_data = []
    prc_data = []
    # 由于不需要进行评价，所以在维持代码正常运行的情况下返回没有意义的值。
    return metric, avg_loss, repres_list, label_list, roc_data, prc_data





def train_test(train_iter, test_iter, config):
    # 加载
    # model = prot_bert.BERT(config)
    # load the saved model
    # model_path = 'Case_AUC=0.894; MCC=0.455; F1=0.473.pl'
    model_path = './rna_model/RNA_AUC=0.843; MCC=0.430; F1=0.513; 1.0-1.0-1.0_27.0_1.pl'
    model = torch.load(model_path)


    print('*=' * 50 + " The model loaded " + '*=' * 50)
    print(model_path)

    if config.cuda:
        model.cuda()

    bert_params = []
    other_params = []
    output_params = []

    # set different lr for BERT and other models
    for name, para in model.named_parameters():
        # print(name)
        if para.requires_grad:
            # if "BERT" in name or "binary_classification" in name or "prot_bert_linear" in name or "pep_bert_linear" in name:
            if "BERT_Embedding" in name:
                bert_params += [para]
            elif "block2" in name:
                output_params += [para]
            else:
                other_params += [para]
    params = [
        {"params": bert_params, "lr": 1e-5},
        {"params": output_params, "lr": 1e-5},
        {"params": other_params, "lr": 1e-5},
    ]

    # optimizer
    optimizer = torch.optim.AdamW(params)

    # Contrastive Residue Loss
    contras_criterion = ContrastiveLoss()

    # Contrastive Type Loss
    contras_type_criterion = ContrastiveTypeLoss()

    # CE Loss
    ce_weight = config.ce_weight
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, ce_weight])).to(config.device)

    # train
    # print('=' * 50 + 'Start Training' + '=' * 50)
    # best_performance = train_model(train_iter, None, test_iter, model, optimizer, criterion, contras_criterion, contras_type_criterion, config, 0)
    # print('=' * 50 + 'Train Finished' + '=' * 50)

    # test
    # print('*' * 60 + 'The Last Test' + '*' * 60)
    print('*=' * 50 + ' The Test Result ' + '*=' * 50)
    # last_test_metric, last_test_loss, last_test_repres_list, last_test_label_list, \
    # last_test_roc_data, last_test_prc_data = model_eval(test_iter, model, criterion, config)
    # lmt = last_test_metric.numpy()

    # 调用model_eval只为了获得预测结果
    model_eval(test_iter, model, criterion, config)

    # print('[ACC,\t\tPrecision,\tSensitivity,\tSpecificity,\tF1,\t\t\tAUC,\t\tMCC,\t\tTP,\t\t\tFP,\t\t\tTN,\t\t\tFN\t]')
    # print('{:.5f}'.format(lmt[0]), '\t{:.5f}'.format(lmt[1]), '\t{:.5f}'.format(lmt[2]), '\t\t{:.5f}'.format(lmt[3]),
    #       '\t\t{:.5f}'.format(lmt[4]), '\t{:.5f}'.format(lmt[5]), '\t{:.5f}'.format(lmt[6]), '\t{:.0f}'.format(lmt[7]),
    #       '\t\t{:.0f}'.format(lmt[8]), '\t\t{:.0f}'.format(lmt[9]), '\t\t{:.0f}'.format(lmt[10]))
    # # print('*' * 60 + 'The Last Test Over' + '*' * 60)
    # return model, best_performance, last_test_metric
    best_performance = 0
    last_test_metric = 0
    return model, best_performance, last_test_metric


def select_dataset():
    path_train_data = train_path
    path_test_data = test_path
    return path_train_data, path_test_data


def load_config():
    '''The following variables need to be actively determined for each training session:
       1.train-name: Name of the training
       2.path-config-data: The path of the model configuration. 'None' indicates that the default configuration is loaded
       3.path-train-data: The path of training set
       4.path-test-data: Path to test set

       Each training corresponds to a result folder named after train-name, which contains:
       1.report: Training report
       2.figure: Training figure
       3.config: model configuration
       4.model_save: model parameters
       5.others: other data
       '''

    '''Set the required variables in the configuration'''
    train_name = 'PepBCL'
    path_config_data = None
    path_train_data, path_test_data = select_dataset()

    '''Get configuration'''
    if path_config_data is None:
        config = cf.get_train_config()
    else:
        config = pickle.load(open(path_config_data, 'rb'))

    '''Modify default configuration'''
    # config.epoch = 50

    '''Set other variables'''
    # flooding method
    b = 0.06

    config.if_multi_scaled = False

    '''initialize result folder'''
    result_folder = '../result/' + config.learn_name
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    '''Save all variables in configuration'''
    config.train_name = train_name
    config.path_train_data = path_train_data
    config.path_test_data = path_test_data

    config.b = b
    # config.if_multi_scaled = if_multi_scaled
    # config.model_name = model_name
    config.result_folder = result_folder

    return config


if __name__ == '__main__':
    np.set_printoptions(linewidth=400, precision=4)
    time_start = time.time()

    '''load configuration'''
    config = load_config()

    '''set device'''
    torch.cuda.set_device(config.device)

    '''load data'''
    train_iter, test_iter = load_data(config)
    print('*=' * 50, ' Load data over', '*=' * 50)
    print("test data set ", test_path)

    '''train procedure'''
    valid_performance = 0
    best_performance = 0
    last_test_metric = 0

    if config.k_fold == -1:
        # train and test
        model, best_performance, last_test_metric = train_test(train_iter, test_iter, config)

    time_end = time.time()
    print('*=' * 50, ' Total time cost ', '*=' * 50)
    print(time_end - time_start, 'seconds')
