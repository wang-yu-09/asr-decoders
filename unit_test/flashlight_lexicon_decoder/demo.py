from fldecoder import LexiconDecoderParams,LexiconDecoder,NBestResult
import numpy as np
import math
import os
dparam = LexiconDecoderParams()

work_dir =  os.path.join( 
            os.path.dirname( 
            os.path.dirname( 
            os.path.dirname( __file__ ) ) ), "unit_test_data" )

dparam.beam_size = 50
dparam.beam_size_token = 50
dparam.beam_threshold = 50
dparam.lm_weight = 1.0
dparam.word_score = -1.0
dparam.unk_score = -math.inf
dparam.sil_score = 0
dparam.log_add = False
dparam.unit_dict = work_dir + "/phones.txt"
dparam.lexicon = "/path/to/word_to_phoneme.txt"
dparam.blank_id = 0
dparam.silence_id = 0
dparam.nbest = 1
dparam.unk = "<unk>"
dparam.kenlm_model = "/path/to/word-3g.bin"

decoder = LexiconDecoder()
res = decoder.init( dparam )
#print( "Return code: ", res )

mat = []
with open(f"{work_dir}/prob_mat.txt","r") as fr:
    for line in fr:
        line = line.strip().split()
        mat.append( [float(r) for r in line])

mat = np.array(mat,dtype="float32")
result = NBestResult()
res = decoder.decode(mat,result)
print( "Decode Return code: ", res )
print( result.units )
print( result.words )

