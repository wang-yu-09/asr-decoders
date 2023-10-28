#!/bin/bash

#################### Configure #####################

# 3rd party
Kenlm_dir=/path/to/Kenlm
Essen_dir=/path/to/Eesen
Openblas_dir=/path/to/Kaldi/tools/OpenBLAS/install

COMPILER=g++
CXX_FLAGS="-O3 -Wall -shared -std=c++11 -DKENLM_MAX_ORDER=6 -fPIC -D_PYBIND11 -Wno-sign-compare"

#################################################

# 0. select lib name

Target_LIST=(
"flashlight"
"rmai"
"eesen"
)
echo "Target:"
for i in "${!Target_LIST[@]}"; do
	echo "$i: ${Target_LIST[$i]}"
done
echo -ne "please input target id:"
read _target_;

if [ "${_target_}" -lt "${#Target_LIST[@]}" ];then
	Target=${Target_LIST[${_target_}]%(*}
else
	echo "invalid target!"
	exit 1
fi

# 1. get some pathes

Src_dir=$(cd `dirname $0`; pwd)/src/${Target}
Out_dir=`pwd`

Python_dir=$(which python | grep anaconda)
Basic_lib_dir=
if [ -n "${Python_dir}" ]; then
    Basic_lib_dir=$(dirname `dirname ${Python_dir}`)
    echo "Found Anaconda: " ${Basic_lib_dir}
else
    Basic_lib_dir=/usr/local
fi

echo "Source Code Dir: " $Src_dir
echo "Work Dir: " $Out_dir

# 2. install pybind11
pb11=`pip list | grep pybind11` 
[ -z "${pb11}" ] && pip install pybind11

# 3. search all cpp files in Src_dir
CPP_fnames=
for cpp_fname in `find ${Src_dir} -name "*.cpp"`; do
    CPP_fnames="${CPP_fnames} ${cpp_fname}" 
done

# 4. compile options
Includes=
Libs=
Out_name=${Target}_decoder

# others
Includes="${Includes} \
         -I${Basic_lib_dir}/include \
         -I${Src_dir}"

if [ "${Target}" == "flashlight" ]; then
    # head files
    Includes="${Includes} -I${Kenlm_dir}"
    # libs
    Libs="${Libs} \
          ${Kenlm_dir}/build/lib/libkenlm.a \
          ${Kenlm_dir}/build/lib/libkenlm_util.a \
          ${Kenlm_dir}/build/lib/libkenlm_builder.a \
          ${Kenlm_dir}/build/lib/libkenlm_filter.a"

elif [ "${Target}" == "rmai" ]; then
    # head files
    Includes="${Includes} -I${Kenlm_dir}"
    # libs
    Libs="${Libs} \
          ${Kenlm_dir}/build/lib/libkenlm.a \
          ${Kenlm_dir}/build/lib/libkenlm_util.a \
          ${Kenlm_dir}/build/lib/libkenlm_builder.a \
          ${Kenlm_dir}/build/lib/libkenlm_filter.a"

elif [ "${Target}" == "eesen" ]; then
    # head files
    Includes="${Includes} \
             -I${Essen_dir}/src \
             -I${Essen_dir}/tools/openfst/include \
             -I${Openblas_dir}/include"
    # libs
    Libs="${Libs} \
          ${Essen_dir}/src/lm/lm.a \
          ${Essen_dir}/src/decoder/decoder.a \
          ${Essen_dir}/src/lat/lat.a \
          ${Essen_dir}/src/cpucompute/cpucompute.a \
          ${Essen_dir}/src/util/util.a \
          ${Essen_dir}/src/base/base.a \
          ${Essen_dir}/src/fstext/fstext.a \
          -Wl,-rpath=${Essen_dir}/tools/openfst/lib \
          -L${Essen_dir}/tools/openfst/lib -lfst \
          -Wl,-rpath=${Openblas_dir}/lib \
          -L${Openblas_dir}/lib -lopenblas -lgfortran \
          -lm -lpthread -ldl"
          
    # 
    CXX_FLAGS="${CXX_FLAGS} -pthread -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN \
               -Wno-sign-compare -Wno-unused-local-typedefs -Winit-self -DHAVE_EXECINFO_H=1 \
               -rdynamic -DHAVE_CXXABI_H -DHAVE_OPENBLAS \
               -Wno-sign-compare -DHAVE_OPENFST_GE_10400 -g"

else
    echo "Unknown Target: " ${Target}
    exit 1;
fi

Out_so_file=${Out_dir}/${Out_name}$(python3-config --extension-suffix)
[ -f ${Out_so_file} ] && rm ${Out_so_file}

# 5. start to compile
${COMPILER} ${CXX_FLAGS} \
${Includes} \
$(python3 -m pybind11 --includes) \
${CPP_fnames} \
-o ${Out_so_file} \
${Libs} \
-L${Basic_lib_dir}/lib \
-L/usr/lib \
-L/usr/local/lib \


if [ -f ${Out_so_file} ]; then
    echo "Build done: " ${Out_so_file}
else
    echo "Build target failed!"
fi