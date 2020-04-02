# @Time    : 2020/3/31 17:35
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : config.py
d =[[6, 128, 'model/pailie3_1.pkl', 1],
    [12, 64, 'model/pailie3_2.pkl', 1],
    [12, 128, 'model/pailie3_3.pkl', 1],
    [12, 128, 'model/pailie3_4.pkl', 2],]
class Config(object):
    NUM_CLASSES = 10
    DATA_SHAPE = (100, 3)
    TIME_STAMP, HIDDEN_SIZE, MODEL_PATH, NUM_LAYERS = d[-1]
    # HIDDEN_SIZE = 64
    DATA_FILE = 'data/pailie3.txt'
    # MODEL_PATH = 'model/pailie3_2.pkl'