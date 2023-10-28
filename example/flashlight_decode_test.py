from flashlight_decoder import DecodeParams,LexiconDecoder,NbestDecodeResult
                           
import numpy as np
import editdistance
from tqdm import tqdm
import time
import sys
import os
import os.path as osp
SOURCE_DIR = osp.dirname(osp.dirname(__file__))
sys.path.insert(0, SOURCE_DIR)
from tools.kaldi_itf import read_matrix_ark
import math

def softmax(X):
    assert(len(X.shape) == 2)
    row_max = np.max(X, axis=1).reshape(-1, 1)
    X -= row_max
    X_exp = np.exp(X)
    s = X_exp / np.sum(X_exp, axis=1, keepdims=True)
    return s

dparam = DecodeParams()

dparam.beam_size = 15
dparam.beam_size_token = 5 # 声学模型剪枝的 beam size, 是number-based pruning
dparam.beam_threshold = 20 # beam剪枝的阈值, 表示当前beam和最佳beam的得分距离, 是score-based pruning
dparam.lm_weight = 0.8
dparam.word_score = 0 # 如果解码得到一个单词, 给它奖励一个得分, log值, 应该<=0 
dparam.unk_score = -math.inf # 如果解码得到一个单词, 给它奖励一个得分, log值, 应该<=0 
dparam.sil_score = 0 # 如果解码得到一个单词, 给它奖励一个得分, log值, 应该<=0 
dparam.log_add = False 
dparam.unit_dict = "/path/to/en-letter.txt"
dparam.lexicon = "/path/to/lexicon_word_to_letters.txt"
dparam.blank_id = 0 
dparam.silence_id = 4 # 如果是CTC的英文字母, 这个ID应该指定为"|"的ID
dparam.nbest = 1
dparam.unk = "<unk>"
dparam.kenlm_model = "/path/to/word-2gram.bin"

decoder = LexiconDecoder()
res = decoder.init( dparam )
print("init lexicon decoder done!")

def read_matrix_npz( prob_fname ):

    ark = np.load( prob_fname )
    data = ark["data"]
    length = ark["length"]
    keys = ark["keys"]

    result = {}
    sidx = 0
    for key,size in zip(keys,length):
        result[key] = data[ sidx:sidx+size, : ]
        sidx += size
    
    return result

prob_fname = "/path/to/acoustic_model_outputs.ark"
probs = read_matrix_ark( prob_fname )

total_edit = 0
total_words = 0
total_frames = 0
total_time = 0

utt2label = {}
with open("archives/test.txt", "r") as fr:
    head = fr.readline()
    for line in fr:
        line = line.strip().split(maxsplit=2)
        utt = line[0]
        label = line[-1]
        utt2label[ utt ] = label

result = DecodeResultNbest()

for t,utt in enumerate(list(probs.keys())):

    #print( f"\n{t}, {utt}" )
    prob = probs[utt]
    start = time.time()

    label = utt2label[utt].split()
    decoder.decode( prob, result )
    res = result.words[0]

    end = time.time()
    total_time += (end-start)
    total_frames += len(prob)
    edit = editdistance.eval( res, label )

    total_edit += edit
    total_words += len(label)

print("CER: ", round( total_edit/total_words*100, 2 ))
print(f"Frames {total_frames} Time { round(total_time,2) } s, Average Time of 1000 frames :", round(total_time*1000/total_frames*1000,4), " ms")



