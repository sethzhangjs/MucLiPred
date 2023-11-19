import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from configuration import config as cf
config = cf.get_train_config()
# print(type(config))
data_root_path = '/home/zjs/PepBCL/data/'
nucleic_type = config.nt
min_seq_len = config.minsl
max_seq_len = config.maxsl
train_path = data_root_path + nucleic_type + '/' + nucleic_type + '_train_' + min_seq_len + '_' + max_seq_len + '.tsv'
test_path = data_root_path + nucleic_type + '/' + nucleic_type + '_test_' + min_seq_len + '_' + max_seq_len + '.tsv'

print(train_path)
# print(params)

# print("nt", params['-nt'])

# nt = params['-nt']
# print(nt)