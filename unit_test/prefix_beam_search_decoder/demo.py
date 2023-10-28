from rmai_decoder import DecodeParams,PrefixBeamSearchDecoder,DecodeResult,DecodeResultNbest
import numpy as np
dparam = DecodeParams()

work_dir = "/path/to/unit_test_data"
dparam.unit_file = "/path/to/phoneme.txt"
dparam.kenlm_file = "/path/to/word-2g.bin"
dparam.lexicon_file = "/path/to/word_to_phoneme.txt"
dparam.prune_beam_size = 5
dparam.prefix_beam_size = 5
dparam.blank_id = 0
dparam.lm_weight = 0.1
#dparam.smooth_factor = 0

decoder = PrefixBeamSearchDecoder()
res = decoder.init( dparam )
#print( "Return code: ", res )

mat = []
with open(f"{work_dir}/prob_mat.txt","r") as fr:
    for line in fr:
        line = line.strip().split()
        mat.append( [float(r) for r in line])

mat = np.array(mat,dtype="float32")

result = DecodeResult()
res = decoder.decode3(mat,result)
print( "Decode Return code: ", res )
print( result.units )

#print( mat.shape )
#print( mat[-1] )

#flip = np.flip( mat,axis=0 ).astype("float32")
#print( flip[0] )
#print( np.flip(mat,axis=0).shape )
#print( flip.min(), flip.max() )
#exit(0)

#result1 = DecodeResult()
#res = decoder.decode( flip, result1 )
#print( result1.units )

