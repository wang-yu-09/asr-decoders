/*
 *  Copyright (c) Streamax Tech. and its affiliates.
 *  Authors:
 *         (1) Wang, Yu  Oct 25, 2023
*/

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <map>

#include "./lm/model.hh"
#include "./util/tokenize_piece.hh"

namespace asrdec
{

float log_add(float probA, float probB);

float log_add(float probA, float probB, float probC);

/* 一个小工具,用于切分长字符串 */
int split_string(const std::string &s, const  std::string seperator, std::vector<std::string> &out_strs);

/* 语言模型得分计算 */
float lm_estimate(std::unique_ptr<lm::ngram::Model> & lm_model,
                    lm::ngram::State & in_state, 
                    std::string & word, 
                    lm::ngram::State & out_state);

/*
* 单元ID索引器
*/
class UnitTable
{
    public:
        UnitTable(){}
        ~UnitTable(){}
    
    public:
        int build(std::string unit_index_file, std::string unk_symbol="<unk>");
        int & idx(std::string sym)
        { /* TODO: 限制边界 */
            if ( m_idxs.find(sym) == m_idxs.end() )
            {
                std::cerr << "Get idx failed! No such symbol in unit table: " << sym << "\n"; 
                return m_idxs[m_unk_symbol];
            }
            return m_idxs[sym]; 
        }
        std::string & sym(int idx)
        { /* TODO: 限制边界 */
            if ( idx < 0 || idx >= m_syms.size() )
            {
                std::cerr << "Get sym failed! Unit table out of range: " << idx << "\n"; 
                return m_unk_symbol;
            }
            return m_syms.at(idx); 
        }
        size_t size(){ return m_syms.size(); }
    
    private:
        std::vector<std::string> m_syms;
        std::unordered_map<std::string,int> m_idxs;
        std::string m_unk_symbol;
};

void softmax(std::vector<float> &inv, size_t length=0);

/*
*/
class LmPruneTable
{
    public:
        LmPruneTable(){}
        ~LmPruneTable(){}
    
    public:
        int build(std::string prune_file, std::shared_ptr<UnitTable> unit_table, size_t max_load_tokens=0);
        std::unordered_map<int,float> & get_suf(int pre_id);
    
    private:
        std::unordered_map<int, std::unordered_map<int,float>> m_prune_table;
};

class PruneTable
{
    public:
        PruneTable(){}
        ~PruneTable(){}

    public:
        int build(std::string lexicon_file, std::shared_ptr<UnitTable> unit_table);
        std::vector<int> & get_suffix(int prefix)
        {
            return m_context[ prefix ];
        }
        std::vector<int> & get_heads(){ return m_heads; }

        int save(std::string fname);
    
    private:
        std::vector<int> m_heads;
        std::unordered_map<int, std::vector<int>> m_context;
        std::shared_ptr<UnitTable> m_unit_table;
};

// class LookAheadTable
// {
//     public:
//         LookAheadTable(){}
//         ~LookAheadTable(){}

//     public:
//         int build(std::string lexicon_file, std::string kenlm_file, std::shared_ptr<UnitTable> unit_table);
//         float get_prob(std::string word, int head_idx)
//         {
//             return m_context[ word ][ head_idx ].first;
//         }
//         int save(std::string fname);
    
//     private:
//         std::map< std::string, 
//                   std::map<int, 
//                            std::pair<float,std::string> > > m_context;
//         std::shared_ptr<UnitTable> m_unit_table;
// };

} // namespace asrdec