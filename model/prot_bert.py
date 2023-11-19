from transformers import BertModel, BertTokenizer
from configuration import config as cf
import re, torch
import torch.nn as nn
import torch.nn.functional as F

cfg = cf.get_train_config()
# 向分类层引入type信息的方式 0: none; 1: add; 2: concat
type_cls_method = cfg.type_cls_method

# block_3 的位置 0: none; 1: inside; 2：outside
block_3_pos = cfg.block_3_pos

# 是否使用block1 1: use; 0: not use
if_block1 = cfg.if_block1

# classificaiton complicate (orginal/simple 0, complicate 1)
clsf_cmplct = cfg.clsf_cmplct

class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()

        global max_len, d_model, device
        # max_len = config.max_len
        d_model = 1024
        device = torch.device("cuda" if config.cuda else "cpu")

        # Server 181
        prot_bert_path = '/home/sde3/wrh/zjs/PepBCL/prot_bert_bfd'

        # BERT
        self.tokenizer = BertTokenizer.from_pretrained(prot_bert_path, do_lower_case=False)
        self.bert = BertModel.from_pretrained(prot_bert_path)

        # Transformer Encoder
        self.encoded_layers = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoded_layers, num_layers=4)

        # ***** FNN after BERT (freeze) *****
        # Block 1 (FNN after Transformer/BERT)
        self.block1 = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 256),
        )

        # ***** Classification Layer *****
        # Block 2_1 (Output module, 256 -> 2)
        self.block2_1 = nn.Sequential(
            # nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Linear(64, 2)
        )

        # Block 2_1_complicated_1 (Output module, 256 -> 2)
        self.block2_1_complicate_1 = nn.Sequential(
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Linear(16, 8),
            nn.ELU(),
            nn.Linear(8, 2)
        )

        # Block 2_1_complicated_2 (Output module, 256 -> 2)
        self.block2_1_complicate_2 = nn.Sequential(
            nn.ELU(),
            nn.Linear(256, 1024),
            nn.ELU(),
            nn.Linear(1024, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 2)
        )

        # Block 2_2 (Output module, 1024 -> 2)
        self.block2_2 = nn.Sequential(
            # nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(1024, 128),
            # nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Linear(64, 2)
        )

        # ***** Block 3 (Dimensionality Reduction for Pooler output) *****
        # Block 3 (for pooler output dimensionality reduction)
        self.block3 = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 256),
        )

        # ***** Block 4 (Dimensionality Reduction for concat) *****
        # Block 4_1 (256 + 1024, 256)
        self.block4_1 = nn.Sequential(
            nn.Linear(256 + 1024, 256),
            nn.ELU()
        )

        # Block 4_2 (256 + 256, 256)
        self.block4_2 = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ELU()
        )

        # Block 4_3 (1024 + 1024, 1024)
        self.block4_3 = nn.Sequential(
            nn.Linear(1024 + 1024, 1024),
            nn.ELU()
        )

    def forward(self, input_seq, type):
        input_seq = ' '.join(input_seq)
        input_seq = re.sub(r"[UZOB]", "X", input_seq)

        encoded_input = self.tokenizer(input_seq, return_tensors='pt')

        type = type.item()
        type_embeddings = torch.zeros_like(encoded_input['input_ids'])

        # move to GPU
        type_embeddings = type_embeddings.cuda()
        for key in encoded_input:
             encoded_input[key] = encoded_input[key].cuda()

        # class_num = 2
        # add type infos to token_type_ids
        if type == 0:
            # DNA type
            encoded_input['token_type_ids'] = torch.zeros_like(encoded_input['input_ids'])
        elif type == 1:
            # RNA type
            encoded_input['token_type_ids'] = torch.ones_like(encoded_input['input_ids'])
        else:
            raise ValueError("type error!")

        # *** BERT Model Output ***
        # class num == 2, we can use BERT Model directly
        bert_output = self.bert(**encoded_input)
        pooler_output = bert_output[1]
        encoder_output = bert_output[0]

        representation = encoder_output.view(-1, 1024)

        # *** Block 1 ***
        if if_block1 == 1:
            # use Block 1
            representation = self.block1(representation)

        # *** Block 3 ***
        # block 3 inside（freeze version）
        if block_3_pos == 1:
            pooler_output = self.block3(pooler_output)  # 1024 -> 256

        return representation, pooler_output


    # get the prediction
    def get_logits(self, x, type):
        # Freeze Bert module
        with torch.no_grad():
            representation, pooler_output = self.forward(x, type)

        # ***** use block 1 or not *****
        if if_block1 == 1:
            # use block 1
            # ***** block 3 inside or outside *****
            if block_3_pos == 1:
                # block 3 inside (freeze version) “11”
                if type_cls_method == 1:
                    # 1.1) add version (256, block3 inside)  ”11_1“
                    representation = representation + pooler_output
                elif type_cls_method == 2:
                    # 1.2) concat version (256, block3 inside) "11_2"
                    repeat_vals = [representation.shape[0] // pooler_output.shape[0]] + [-1] * (len(pooler_output.shape) - 1)
                    representation = torch.cat((representation, pooler_output.expand(*repeat_vals)), dim=-1)
                    representation = representation.view(-1, 256 + 256)
                    representation = self.block4_2(representation)  # 256 + 256 -> 256
                else:
                    # 1.3) none version (256, block3 inside) "11_0"
                    # (block 3 inside) only for type contrastive learning
                    pass
            elif block_3_pos == 2:
                # block 3 outside (unfreeze version) "12"
                if type_cls_method == 1:
                    # 2.1) add version (1024, block3 outside) "12_1"
                    pooler_output = self.block3(pooler_output)  # 1024 -> 256
                    representation = representation + pooler_output
                elif type_cls_method == 2:
                    # 2.2) concat version (1024, block3 outside) "12_2"
                    # representation = representation.view(-1, 256 + 1024)  # 256 + 1024 -> 256
                    # representation = self.block4_1(representation)
                    pooler_output = self.block3(pooler_output)  # 1024 -> 256
                    repeat_vals = [representation.shape[0] // pooler_output.shape[0]] + [-1] * (len(pooler_output.shape) - 1)
                    representation = torch.cat((representation, pooler_output.expand(*repeat_vals)), dim=-1)
                    representation = representation.view(-1, 256 + 256)
                    representation = self.block4_2(representation)  # 256 + 256 -> 256
                else:
                    # 2.3) none version (1024, block3 inside) "12_0"
                    raise ValueError('use block 3 outside but not use type_cls_method')
            else:
                # 使用block1，不使用block 3 -> 不向分类层引入type信息 "10"
                if type_cls_method != 0:
                    raise ValueError('use block 1 but not use block 3, cannot use type_cls_method')  # because of different dimensionalities
        else:
            # not use block 1
            # pooler_output and encoder_output have same dimensionality -> no need to do dimensionality reduction
            # -> no need to use block 3 -> change the input dimensionality of the classification layer
            # -> use block 2_2(1024 -> 2)
            if block_3_pos == 0:
                # without block 1 and block 3; bert_output 1024; pooler_output 1024 "00"
                if type_cls_method == 1:
                    # 3.1) add version "00_1"
                    representation = representation + pooler_output  # 1024
                elif type_cls_method == 2:
                    # 3.2) concat version "00_2"
                    repeat_vals = [representation.shape[0] // pooler_output.shape[0]] + [-1] * (len(pooler_output.shape) - 1)
                    representation = torch.cat((representation, pooler_output.expand(*repeat_vals)), dim=-1)
                    representation = representation.view(-1, 1024 + 1024)
                    representation = self.block4_3(representation)  # 1024 + 1024 -> 1024
                else:
                    # 3.3) none "00_0"
                    pass
            else:
                # not use block 1 -> no need to do dimensionality reduction -> no need to use block 3
                # but use block 3 -> error
                raise ValueError('without block 1, no need to use block 3')

        # ***** get logits *****
        if if_block1 == 1:
            # use Block 1 -> use block 2_1 (256 -> 2)
            logits = self.block2_1(representation)  # 256 -> 2
        else:
            # not use Block 1 -> use block 2_2 (1024 -> 2)
            logits = self.block2_2(representation)  # 1024 -> 2

        return logits

