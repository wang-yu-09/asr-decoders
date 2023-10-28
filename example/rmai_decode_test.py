from rmai_decoder import ( DecodeParams,GreedyDecoder,DecodeResult,UnitTable,
                           PrefixBeamSearchDecoder,LexiconTransducer,
                           DecodeResultNbest)
                           
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

def softmax(X):
    assert(len(X.shape) == 2)
    row_max = np.max(X, axis=1).reshape(-1, 1)
    X -= row_max
    X_exp = np.exp(X)
    s = X_exp / np.sum(X_exp, axis=1, keepdims=True)
    return s

dparam = DecodeParams()
dparam.unit_file = "/path/to/en-letter.txt"
dparam.unit_kenlm_file = "/path/to/letter-3g.bin"
dparam.lexicon_file = "/path/to/lexicon_word_to_letters.txt"
dparam.wwi_kenlm_file = "/path/to/headletter-3g.bin"
dparam.unk_symbol = "<unk>"
dparam.unit_beam_size = 5
dparam.beam_size = 15
dparam.blank_id = 0
dparam.silence_id = 0
dparam.unit_lm_weight = 0.8 #0.02
dparam.wwi_lm_weight = 0.2 #0.02
dparam.max_silence_frames = 5
dparam.callback_pruned_beam = False
dparam.truncate_silence = False
dparam.word_kenlm_file = "/path/to/word-2g.bin"
dparam.word_lm_weight = 0.01

pdecoder = PrefixBeamSearchDecoder()
pdecoder.init( dparam )
print("init prefix beam search decoder done!")

def read_matrix_npz(prob_fname):

    ark = np.load(prob_fname)
    data = ark["data"]
    length = ark["length"]
    keys = ark["keys"]

    result = {}
    sidx = 0
    for key,size in zip(keys,length):
        result[key] = data[sidx:sidx+size, :]
        sidx += size
    
    return result

prob_fname = "/path/to/acoustic_model_outputs.ark"
probs = read_matrix_ark(prob_fname)

total_edit = 0
total_words = 0
total_frames = 0
total_time = 0

result = DecodeResult()

unittable = UnitTable()
unittable.build(dparam.unit_file)

gdecoder = GreedyDecoder()
gdecoder.init(dparam)
print("init greedy search decoder done!")

dparam.word_beam_size = 15
dparam.lexicon_file = "/path/to/lexicon_word_to_letters.txt"
dparam.word_kenlm_file = "/path/to/word-2g.bin"

transducer = LexiconTransducer()
transducer.init( dparam )
print("init lexicon transducer done!")

utt2label = {}
with open("archives/test.txt", "r") as fr:
    head = fr.readline()
    for line in fr:
        line = line.strip().split(maxsplit=2)
        utt = line[0]
        label = line[-1]
        utt2label[ utt ] = label

for t,utt in enumerate(tqdm(list(probs.keys()))):

    #print( f"\n{t}, {utt}" )
    prob = probs[utt]
    start = time.time()

    label = utt2label[utt].split()

    pdecoder.decode( prob, result, 0 )

    res_u = list( ("".join( result.units )).replace("|", "") )
    if transducer.decode( res_u, result ) == False:
        continue
    res = [ w for w in result.words if w != '<unk>' ]

    end = time.time()
    total_time += (end-start)
    total_frames += len(prob)
    edit = editdistance.eval(res, label)

    total_edit += edit
    total_words += len(label)

print("CER: ", round( total_edit/total_words*100, 2 ) )
print(f"Frames {total_frames} Time { round(total_time,2) } s, Average Time of 1000 frames :", round(total_time*1000/total_frames*1000,4), " ms")



