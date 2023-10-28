import numpy as np
import os
project_dir = os.path.dirname( os.path.abspath(__file__) )

mat = []
with open( f"{project_dir}/mat.txt", "r") as fr:
    for line in fr:
        line = line.strip().split()
        mat.append( [ float(t) for t in line ] )

mat = np.array(mat, dtype="float32")
print( mat.shape )

id2phones = {}
with open( f"{project_dir}/phones.txt", "r") as fr:
    idx = 0
    for line in fr:
        line = line.strip()
        if len(line) == 0:
            continue
        id2phones[ idx ] = line
        idx += 1

print( len(id2phones) )

ids = np.argmax( mat, axis=1 )
first_pass = []
for fid,idx in enumerate(ids):
    if len(first_pass) == 0 or idx != first_pass[-1][1]:
        first_pass.append( (fid,idx) )
print( first_pass )

for fid,idx in first_pass:
    if idx != 0:
        print( "fid: ", fid, " unit: ", id2phones[idx] )
print()