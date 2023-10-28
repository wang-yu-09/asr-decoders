/*
 *  Copyright (c) Streamax Tech. and its affiliates.
 *  Authors:
 *         (1) Wang, Yu  Oct 25, 2023
*/
#pragma once

#include <string>
#include <memory>
#include <limits>
#include <vector>

#include "decoder/lattice-faster-decoder.h"
#include "fstext/fstext-lib.h"

/* 注意: 虽然这个解码器不返回lattice, 但实际上我是包装的是Kaldi的 latgen-faster-decoder 
 *       而不是 faster-decoder
*/

#ifdef _PYBIND11
/* Python Interface 
 * 这个python接口是基于pybind的，如果你想使用这个python接口
 * 1. 安装pybind: pip install pybind11
 * 2. 编译: 参考我给的编译指令
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
#endif

namespace tlg
{
    /* 解码器配置 */
    struct DecodeParams
    {   
        // TLG.fst 解码图
        std::string fst_in;
        std::string word_file;

        float beam = 16.0;
        int max_active = std::numeric_limits<int32>::max();
        int min_active = 200;
        float lattice_beam = 10.0;
        int prune_interval = 25;
        bool determinize_lattice = true;
        float beam_delta = 0.5;
        float hash_ratio = 2.0;
        float prune_scale = 0.1;
        float acoustic_scale = 0.1;
        float allow_partial = false;
    };

    /* 解码结果 */
    struct LinearDecodeResult
    {
        std::vector<std::string> words;
        std::vector<int> alignments;
        float am_score;
        float lm_score;

        public:
            void clear()
            {
                words.clear();
                alignments.clear();
                am_score = 0.0;
                lm_score = 0.0;
            }
    };
    
    /* 解码器, 解码结果直接就输出单词而不是lattice */
    class FasterDecoder
    {
        public:
            FasterDecoder()
            {
                m_decoder_backend.reset(nullptr);
                m_symbol_table.reset(nullptr);
            }

            ~FasterDecoder()
            {
                if (m_decoder_backend != nullptr)
                {
                    m_decoder_backend.reset(nullptr);
                }
                if (m_symbol_table != nullptr)
                {
                    m_symbol_table.reset(nullptr);
                }
            }
        
        public:
            int init(DecodeParams &dparam);
            int decode(const float *prob_in, size_t T, size_t D, LinearDecodeResult &result_out);
        
            /*Python Interface*/
            #ifdef _PYBIND11
            int pydecode(const py::array_t<float> &prob_in, LinearDecodeResult &result_out);
            #endif

        private:
            /* 备份配置列表 */
            DecodeParams m_dparam;
            /* Kaldi LatgenFasterDecoder backend */
            std::unique_ptr<eesen::LatticeFasterDecoder> m_decoder_backend;
            /* 单词表 */
            std::unique_ptr<fst::SymbolTable> m_symbol_table;

    };

} // namespace tlg