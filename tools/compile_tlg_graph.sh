#!/bin/bash

# Authors: 
#   (1) Wang Yu, Oct 28, 2023

# This script compiles the ARPA-formatted language models into FSTs. Finally it composes the LM, lexicon
# and token FSTs together into the decoding graph. 

Project_dir=`pwd`
export PATH=$PATH:${Project_dir}/../../tools/openfst/bin:${Project_dir}/../../src/fstbin

if [ $# -ne 3 ]; then
  echo "Usage: $0 <lang-dir> <lm> <graph-dir>"
  echo "e.g.: compile_tlg_graph.sh lang lm/3-gram.arpa graph"
  exit 1
fi

langdir=$1
lm=$2
graphdir=$3
mkdir -p $graphdir

echo "This operation may take some time ... "
cp ${langdir}/words.txt $graphdir || exit 1;

# gunzip -c $lmdir/lm_${lm_suffix}.arpa.gz | utils/find_arpa_oovs.pl $test/words.txt  > $tmpdir/oovs_${lm_suffix}.txt
cat $lm | utils/find_arpa_oovs.pl $graphdir/words.txt  > $graphdir/oovs.txt

# grep -v '<s> <s>' because the LM seems to have some strange and useless
# stuff in it with multiple <s>'s in the history.
cat $lm | \
grep -v '<s> <s>' | \
grep -v '</s> <s>' | \
grep -v '</s> </s>' | \
arpa2fst - | fstprint | \
utils/remove_oovs.pl $graphdir/oovs.txt | \
utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$graphdir/words.txt \
--osymbols=$graphdir/words.txt  --keep_isymbols=false --keep_osymbols=false | \
fstrmepsilon | fstarcsort --sort_type=ilabel > $graphdir/G.fst
    
# Compose the final decoding graph. The composition of L.fst and G.fst is determinized and
# minimized.
fsttablecompose ${langdir}/L.fst $graphdir/G.fst | fstdeterminizestar --use-log=true | \
fstminimizeencoded | fstarcsort --sort_type=ilabel > $graphdir/LG.fst || exit 1;
fsttablecompose ${langdir}/T.fst $graphdir/LG.fst > $graphdir/TLG.fst || exit 1;

echo "Composing decoding graph TLG.fst succeeded"
