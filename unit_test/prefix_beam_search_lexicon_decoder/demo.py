from prefix_beam_search_lexicon_decoder import DecodeParams,PrefixBeamSearchLexiconDecoder,DecodeResult
import numpy as np
dparam = DecodeParams()

work_dir = "/path/to/test_data"
dparam.lexiconp_file = work_dir + "/lexiconp.txt"
dparam.unit_file = work_dir + "/phones.txt"
dparam.unk_symbol = "<unk>"
dparam.kenlm_file = "/path/to/word-2g.bin"
dparam.prune_beam_size = 5
dparam.word_beam_size = 5
dparam.prefix_beam_size = 5
dparam.blank_id = 0
dparam.lm_weight = 0.1

decoder = PrefixBeamSearchLexiconDecoder()
res = decoder.init( dparam )
#print( "Return code: ", res )

mat = []
with open(f"{work_dir}/prob_mat.txt","r") as fr:
    for line in fr:
        line = line.strip().split()
        mat.append( [float(r) for r in line])

mat = np.array(mat,dtype="float32")
result = DecodeResult()
res = decoder.decode(mat,result)
print( "Decode Return code: ", res )
print( result.words )

