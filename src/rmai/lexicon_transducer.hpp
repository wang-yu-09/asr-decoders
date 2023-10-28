/*
 *  Copyright (c) Streamax Tech. and its affiliates.
 *  Authors:
 *         (1) Wang, Yu  Oct 25, 2023
*/
#pragma once
#include <string>
#include <vector>
#include <memory>

#include "decoder_arks.hpp"
#include "decoder_utils.hpp"
#include "lexicon_trie.hpp"

namespace asrdec{
namespace lt
{
    struct PassToken
    {   
        /* 指向输入序列的索引位置 */
        int frame_id;
        /* 指向树的节点指针 */
        std::shared_ptr<asrdec::lt::TrieNode> trie_node = nullptr;
        /* 解码得到的单词序列 */
        std::vector<std::string> words;
        /* 该单词序列的语言模型得分 */
        float score = 0.0;
        /* 语言模型状态 */
        lm::ngram::State lm_state;
        /* 对齐 */
        // vector<string> units;

        public:
            PassToken(){}
            ~PassToken(){}
            PassToken(int frame_id_, std::shared_ptr<asrdec::lt::TrieNode> trie_node_): \
                    frame_id(frame_id_), trie_node(trie_node_){}
    };

    class LexiconTransducer
    {
        public:
            LexiconTransducer(){}
            ~LexiconTransducer(){}

        public:
            int init(asrdec::DecodeParams &dparam);
            //int decode(const std::vector<std::string> & units, std::vector<std::string> &result_out);
            float decode(const std::vector<std::string> & units, asrdec::DecodeResult &result_out);
            
        private:
            double lm_estimate(std::vector<std::string> &words);

        private:
            asrdec::DecodeParams m_dparam;
            std::unique_ptr<asrdec::lt::LexiconTrie> m_lexicon;
            std::unique_ptr<lm::ngram::Model> m_kenlm_model;
            std::shared_ptr<asrdec::UnitTable> m_unit_table;
    };

} // namespace lt

} // namespace asedec