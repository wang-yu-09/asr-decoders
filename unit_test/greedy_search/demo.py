from rmai_decoder import DecodeParams,GreedyDecoder,DecodeResult
import numpy as np
dparam = DecodeParams()

work_dir = "/path/to/unit_test_data"
dparam.unit_file = work_dir + "/phones.txt"
dparam.blank_id = 0

decoder = GreedyDecoder()
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
print("Decode Return code: ", res)
print(result.units)

