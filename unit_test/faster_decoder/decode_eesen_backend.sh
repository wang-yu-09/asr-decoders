export PATH=$PATH:/path/to/Eesen/src/decoderbin

field=dev_liv,dev_stv
decode_max_active=5000
decode_min_active=200
decode_max_mem=50000000
decode_beam=17.0
lattice_beam=8.0
decode_acwt=0.9
# 下面这些参数为了处理lattice
word_ins_penalty=0.0,0.5,1.0,1.5,2.0
min_acwt=1
max_acwt=20
acwt_factor=0.05

latgen-faster \
--verbose=1 \
--max-active=$decode_max_active \
--min-active=$decode_min_active \
--max-mem=$decode_max_mem \
--beam=$decode_beam \
--lattice-beam=$lattice_beam \
--acoustic-scale=$decode_acwt \
--allow-partial=true \
--word-symbol-table=/path/to/graph/words.txt \
/path/to/graph/TLG.fst \
ark:test.ark \
ark:test.lat