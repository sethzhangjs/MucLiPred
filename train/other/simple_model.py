from transformers import BertModel, BertTokenizer

import re, torch
import torch.nn as nn
import torch.nn.functional as F


class Guide(nn.Module):
    def __init__(self, config):
        super(Guide, self).__init__()

        global max_len, d_model, device
        d_model = 1024
        device = torch.device("cuda" if config.cuda else "cpu")

        # Server 181
        prot_bert_path = '/home/sde3/wrh/zjs/PepBCL/prot_bert_bfd'

        # BERT
        self.tokenizer = BertTokenizer.from_pretrained(prot_bert_path, do_lower_case=False)
        self.bert = BertModel.from_pretrained(prot_bert_path)

        # BERT Embedding
        self.BERT_Embedding = self.bert.embeddings

        # Manual Word Embedding
        self.word_embedding = nn.Embedding(9, 1024)

        # Manual Type Embedding
        self.type_embedding = nn.Embedding(2, 1024)

        # Transformer Encoder
        self.encoded_layers = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoded_layers, num_layers=4)

        # Block 1 (FNN)
        self.block1 = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 256),
        )

        # Block 2 (Output module)
        self.block2 = nn.Sequential(
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


    def forward(self, input_seq, type):
        input_seq = ' '.join(input_seq)
        input_seq = re.sub(r"[UZOB]", "X", input_seq)
        encoded_input = self.tokenizer(input_seq, return_tensors='pt')

        # get the type and initialize
        type = type.item()
        type_embeddings = torch.zeros_like(encoded_input['input_ids'])
        word_embeddings = torch.zeros_like(encoded_input['input_ids'])

        # move to GPU
        word_embeddings = word_embeddings.cuda()
        type_embeddings = type_embeddings.cuda()
        for key in encoded_input:
             encoded_input[key] = encoded_input[key].cuda()

        # word embedding
        input_ids = encoded_input['input_ids']
        word_embeddings = self.word_embedding(input_ids)

        # type Embedding
        if type == 0:
            # print('DNA type')
            type_embeddings = self.type_embedding(torch.zeros_like(encoded_input['input_ids']))
        elif type == 1:
            # print('RNA type')
            type_embeddings = self.type_embedding(torch.ones_like(encoded_input['input_ids']))
        else:
            print("type error!")
            print("current type is: ", type)

        # word_embeddings + type_embeddings
        input_embeddings = word_embeddings + type_embeddings

        # *** Transformer Encoder ***
        output = self.transformer_encoder(input_embeddings)

        return output


    # get the prediction
    def get_logits(self, x, type):
        # freeze Bert module
        with torch.no_grad():
            output = self.forward(x, type)
        representation = output.view(-1, 1024)
        output = self.block1(representation)
        logits = self.block2(output)
        return logits

