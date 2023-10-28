cmake_minimum_required(VERSION 3.4.1)

project(demo)

set(CMAKE_CXX_STANDARD 11)
set(KENLM_ROOT /path/to/Kenlm)
add_definitions(-DKENLM_MAX_ORDER=6)

add_executable(test_demo 
    ${CMAKE_CURRENT_SOURCE_DIR}/demo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/rmai/decoder_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/rmai/lexicon_trie.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/rmai/lexicon_transducer.cpp
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/rmai
    ${KENLM_ROOT}
    )       

target_link_libraries(
                    test_demo
                    ${KENLM_ROOT}/build/lib/libkenlm.a
                    ${KENLM_ROOT}/build/lib/libkenlm_util.a
                    /home/wy_rmai/anaconda3/lib/liblzma.a
                    /home/wy_rmai/anaconda3/lib/libbz2.a
                    /home/wy_rmai/anaconda3/lib/libz.a
)