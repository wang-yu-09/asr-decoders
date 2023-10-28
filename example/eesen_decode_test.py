from eesen_decoder import DecodeParams,FasterDecoder,LinearDecodeResult

import numpy as np
import editdistance
from tqdm import tqdm
import time

def softmax(X):
    assert(len(X.shape) == 2)
    row_max = np.max(X, axis=1).reshape(-1, 1)
    X -= row_max
    X_exp = np.exp(X)
    s = X_exp / np.sum(X_exp, axis=1, keepdims=True)
    return s


dparam = DecodeParams()

best_wer = None

ss = [1,3,5,10]#,20,30,40,50,60]
ss.reverse()
for amscale in ss:

    print( ">>>>>>>>> amscale:", amscale )

    dparam.fst_in = "/path/to/TLG.fst"
    dparam.word_file = "/path/to/words.txt"
    dparam.beam = 13.0
    dparam.max_active = 5000
    dparam.min_active = 200
    dparam.lattice_beam = 8.0
    dparam.prune_interval = 25
    dparam.determinize_lattice = True
    dparam.beam_delta = 0.5
    dparam.hash_ratio = 2.0
    dparam.prune_scale = 0.1
    dparam.acoustic_scale = amscale
    dparam.allow_partial = True

    decoder = FasterDecoder()
    res = decoder.init(dparam)

    print("Init Decoder Done!")

    prob_fname = "/path/to/acoustic_model_outputs.npz"
    ark = np.load(prob_fname)
    probs = ark["prob"]
    labels = ark["label"]
    utts = ark["uttID"]

    total_edit = 0
    total_words = 0
    total_frames = 0
    total_time = 0

    result = LinearDecodeResult()

    for i in tqdm(range(len(labels))):
    #for prob, label in tqdm( zip( probs, labels ) ):

        prob = np.log(softmax(probs[i]))
        label = labels[i]
        utt = utts[i]
        start = time.time()
        
        decoder.decode(prob, result)

        end = time.time()
        total_time += (end-start)
        total_frames += len(prob)

        char_label = utt2chars[utt]

        edit = editdistance.eval(result.words, char_label)
        total_edit += edit
        total_words += len(char_label)

    cer = round( total_edit/total_words*100, 2)
    print("CER: ", cer )
    print(f"Frames {total_frames} Time {total_time} Average Time of 1000 frames :", total_time*1000/total_frames)
    if best_wer is None:
        best_wer = (amscale, cer)
    elif cer < best_wer[1]:
        best_wer = (amscale, cer)

print( "####################" )
print( "Best CER:", best_wer )
print( "####################" )


