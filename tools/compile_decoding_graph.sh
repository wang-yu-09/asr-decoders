#!/bin/bash

# authors: 
#       (1) wang yu, Oct 28, 2023
#

Eesen_root=/home/wy_rmai/Project/Eesen2
Word_dir=$(cd `dirname $0`; pwd)
stage=1
lm=3-gram.arpa
label= # data/finetune_5h_label.int

if [ $stage -le 1 ]; then
    echo "====================================="
    echo " Compiling T.fst and L.fst"
    echo "====================================="
    # 编译 T 和 L WFST
    ${Word_dir}/utils/ctc_compile_dict_token.sh --dict-type "phn" ${Word_dir}/src ${Word_dir}/lang/temp ${Word_dir}/lang || exit 1;
fi

if [ $stage -le 2 ]; then
    echo "====================================="
    echo " Compiling G.fst and TLG.fst"
    echo "====================================="
    # 编译 G WFST，并且合成TLG解码图
    ${Word_dir}/compile_tlg_graph.sh ${Word_dir}/lang ${Word_dir}/${lm} ${Word_dir}/graph || exit 1;
fi

if [ $stage -le 3 && -n ${label} ]; then
    echo "================================================"
    echo " Computing the statistics of training labels"
    echo "================================================"
    # 从训练数据的对齐中获得计算先验概率统计量
    mkdir -p graph/log
    cat $label | awk '{line=$0; gsub(" "," 0 ",line); print line " 0";}' | \
        ${Eesen_root}/src/decoderbin/analyze-counts --verbose=1 --binary=false ark:- ${Word_dir}/graph/label.counts >& ${Word_dir}/graph/log/compute_label_counts.log || exit 1
fi