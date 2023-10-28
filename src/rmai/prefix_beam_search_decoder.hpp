/*
 *  Copyright (c) Streamax Tech. and its affiliates.
 *  Authors:
 *         (1) Wang, Yu  Oct 25, 2023
*/
#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <map>
#include <limits>

#include "decoder_utils.hpp"
#include "decoder_arks.hpp"
#include "base_decoder.hpp"
// #include "hotword_graph.hpp"
#include "lexicon_trie.hpp"

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

namespace asrdec {
namespace pbsd
{
    const float NEG_INF = -std::numeric_limits<float>::infinity();

    struct WordState
    {
        // 词典树节点指针
        std::shared_ptr<asrdec::lt::TrieNode>  lexicon_trie_node;
        // 词间语言模型得分
        float wwi_score = 0.0;
        // 解码得到的单词数量
        size_t n_words = 0;
        // 词间语言模型状态
        lm::ngram::State wwi_lm_state;
        // 词尾语言模型得分及状态
        float word_score = 0.0;
        lm::ngram::State word_lm_state;

        public:
            WordState( std::shared_ptr<asrdec::lt::TrieNode> _trie_node,
                       float _wwi_score,
                       size_t _n_words,
                       lm::ngram::State _wwi_lm_state
                     ): lexicon_trie_node(_trie_node),
                        wwi_score(_wwi_score),
                        n_words(_n_words),
                        wwi_lm_state(_wwi_lm_state){}

            WordState( std::shared_ptr<asrdec::lt::TrieNode> _trie_node,
                       float _wwi_score,
                       size_t _n_words,
                       lm::ngram::State _wwi_lm_state,
                       float _word_score,
                       lm::ngram::State _word_lm_state
                     ): lexicon_trie_node(_trie_node),
                        wwi_score(_wwi_score),
                        n_words(_n_words),
                        wwi_lm_state(_wwi_lm_state),
                        word_score(_word_score),
                        word_lm_state(_word_lm_state){}

    };

    /* 前缀树搜索状态保存 */
    struct PrefixBeam
    {   
        /* blank前缀的概率 */
        float prob_b = NEG_INF;
        /* 非blank前缀的概率 */
        float prob_nb = NEG_INF;
        /* unit级别语言模型概率: 如果使用语言模型, 这个得分为最佳解码单词序列 */
        float prob_lm = 0.0;
        /* 这条beam的总得分 */ 
        float prob_total = NEG_INF; 
        /* unit级别语言模型状态 */
        lm::ngram::State lm_state; 

        /* 词典树状态 */ 
        std::vector< std::shared_ptr<asrdec::lt::TrieNode> > lexicon_trie_nodes;
        /* 词间语言模型 */
        std::vector<WordState> word_states;

        public:
            explicit PrefixBeam(float prob_b_, 
                                float prob_nb_, 
                                float prob_lm_, 
                                float prob_total_
                                ):prob_b(prob_b_),
                                prob_nb(prob_nb_),
                                prob_lm(prob_lm_),
                                prob_total(prob_total_)
                                { }
                explicit PrefixBeam(){}

        inline void show()
        {
            std::cout << "prob_b: " << prob_b << " prob_nb: " << prob_nb << " prob_lm: " << prob_lm << " total: "<< prob_total << std::endl;
        }

    };

    /* 前缀树搜索 + 单元级别语言模型 
     */
    class PrefixBeamSearchDecoder: public asrdec::bd::BaseDecoder
    {
        public:
            PrefixBeamSearchDecoder(){}
            ~PrefixBeamSearchDecoder(){}
        
        public:
            int init(DecodeParams &dparam);

            int decode(const float *prob_in, size_t T, size_t D, DecodeResult &result_out)
            {   
                std::cerr << ".decode has been over written by decode[0-3], please use them!\n";
                return -1;
            }

            /* 0: 最基本的前缀束解码算法 */
            int decode0(const float *prob_in, size_t T, size_t D, DecodeResult &result_out);
            /* 1: 0 + 支持跳帧, 热词, 剪枝召回 */
            // 20231028：本代码中不支持热词解码
            int decode1(const float *prob_in, size_t T, size_t D, DecodeResult &result_out);
            /* 2: 1 + 词典树剪枝 */
            int decode2(const float *prob_in, size_t T, size_t D, DecodeResult &result_out);
            /* 3: 2 + 词间语言模型 */
            int decode3(const float *prob_in, size_t T, size_t D, DecodeResult &result_out);
            /* 4: 3 + 单词语言模型 */
            int decode4(const float *prob_in, size_t T, size_t D, DecodeResult &result_out);

            /*Python Interface*/
            #ifdef _PYBIND11
            int pydecode_mode(const py::array_t<float> &prob_in, asrdec::DecodeResult &result_out, int mode);
            #endif

        private:           
            // int decode_nbest(const float *prob_in, size_t T, size_t D, DecodeResultNbest &result_out);

            int init_new_prefix_beam(       PrefixBeam &prev_beam, 
                std::map<std::vector<int>, PrefixBeam> &new_beams, 
                                      std::vector<int> &new_prefix, 
                std::map<std::vector<int>, PrefixBeam> &pruned_beams,
                                           const float *prob_in,
                                                size_t &T, 
                                                size_t &D,
                                                size_t &t,
                                                 float &unit_prob,
                                                size_t &unit_id);

        private:
            DecodeParams m_dparam;
            std::unique_ptr<lm::ngram::Model> m_unit_kenlm_model;
            std::shared_ptr<asrdec::UnitTable> m_unit_table;
            float m_unit_unk_lm_score = 0.0;

            /* 实验性的配置 */
            // std::unique_ptr<asrdec::hg::HotwordGraph> m_hotword_graph;
            std::unique_ptr<asrdec::lt::LexiconTrie> m_lexicon_trie;
            std::unique_ptr<lm::ngram::Model> m_wwi_kenlm_model;
            float m_wwi_unk_lm_score = 0.0;
            std::unique_ptr<lm::ngram::Model> m_word_kenlm_model;
            float m_word_unk_lm_score = 0.0;
    };

    void smooth(std::vector<float> &probin, int factor=0.25);

} // namespace pbsd
} // namespace asrdec