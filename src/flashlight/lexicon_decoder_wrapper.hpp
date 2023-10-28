/*
 *  Copyright (c) Streamax Tech. and its affiliates.
 *  Authors:
 *         (1) Wang, Yu  Oct 25, 2023
 */
#pragma once

#include "decoder/LexiconDecoder.h"
#include "decoder/Trie.h"
#include "decoder/lm/LM.h"
#include "dictionary/Dictionary.h"

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

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

namespace flashlight
{
    struct DecodeParams
    {
        int beamSize; 
        int beamSizeToken;
        double beamThreshold;
        double lmWeight;
        double wordScore;
        double unkScore;
        double silScore;
        bool logAdd;
        std::string unit_dict; // 子词词典
        std::string lexicon; // 发音词典
        int blank_id; 
        int silence_id;
        int nbest;
        std::string unk;
        std::string kenlm_model;
    };

    struct NbestDecodeResult
    {
        int size;
        std::vector<std::vector<std::string> > words;
        std::vector<std::vector<std::string> > units;

        public:
            void clear()
            {
                size = 0;
                words.clear();
                units.clear();
            }
    };

    class LexiconDecoderWrapper
    {
        public:
            LexiconDecoderWrapper()
            {
                m_decode_backend.reset( nullptr );
            }

            ~LexiconDecoderWrapper()
            {
                if ( m_decode_backend != nullptr )
                {
                    m_decode_backend.reset( nullptr );
                }
            }
        
        public:
            int init(DecodeParams & dparam);

            int decode(const float *prob_in, size_t T, size_t D, NbestDecodeResult &result_out);

            /*Python Interface*/
            #ifdef _PYBIND11
            int pydecode(const py::array_t<float> &prob_in, NbestDecodeResult &result_out);
            #endif

        private:
            /* 子词词典 */ 
            std::unordered_map<std::string,int> m_unit_dict;
            std::vector<std::string> m_units;
            Dictionary m_word_dict;
            /* 单词级别语言模型 */
            LMPtr m_lm; // shared指针
            /* 词典树 */
            TriePtr m_trie; // shared指针
            /* 其他 */
            std::vector<float> m_transitions;
            /* flashlight解码器后端 */
            std::unique_ptr<LexiconDecoder> m_decode_backend;
            /* 参数备份 */
            DecodeParams m_dparam;
    };

}