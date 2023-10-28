from eesen_decoder import DecodeParams,FasterDecoder,LinearDecodeResult
import numpy as np
import math
import os
import math
import sys
import time
import os.path as osp
SOURCE_DIR = osp.dirname(osp.dirname(osp.dirname(__file__)))
sys.path.insert(0, SOURCE_DIR)
from tools.kaldi_itf import read_matrix_ark

dparam = DecodeParams()

dparam.fst_in = "/path/to/graph/TLG.fst"
dparam.word_file = "/path/to/graph/words.txt"
dparam.beam = 13
dparam.max_active = 5000
dparam.min_active = 200
dparam.lattice_beam = 8
dparam.prune_interval = 25
dparam.determinize_lattice = True
dparam.beam_delta = 0.5
dparam.hash_ratio = 2.0
dparam.prune_scale = 0.1
dparam.acoustic_scale = 0.1
dparam.allow_partial = True

decoder = FasterDecoder()
res = decoder.init( dparam )

print( "init decoder done!" )

mat = []
with open(f"../../unit_test_data/prob_mat.txt","r") as fr:
    for line in fr:
        line = line.strip().split()
        mat.append( [float(r) for r in line])

mat = np.log(np.array(mat,dtype="float32"))

#write_matrix_ark( {"test":mat}, "test.ark" )

#print( mat[0] )
#print( np.sum(mat[0]) )
#exit(0)
#print( mat.shape )

result = LinearDecodeResult()
start = time.time()
res = decoder.decode(mat, result)
end = time.time()
print( end-start )
print( "Decode Return code: ", res )
print( result.words )

