/*
 *  Copyright (c) Streamax Tech. and its affiliates.
 *  Authors:
 *         (1) Wang, Yu  Oct 25, 2023
*/

#pragma once

#include <vector>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>

#include "decoder_utils.hpp"

namespace asrdec {
namespace lt
{

const double kMinusLogThreshold = -39.14;

enum class SmearingMode 
{
    NONE = 0,
    MAX = 1,
    LOGADD = 2,
};

/*
    * TrieNode is the trie node structure in Trie.
    */
struct TrieNode
{   
    // child symbol_idx to the child node
    std::map<int, std::shared_ptr<TrieNode> > children;
    // Labels of words that are constructed from the given path
    int symbol_idx;
    std::vector<std::string> labels;
    std::vector<float> scores;
    // Maximum score of all the labels if this node is a leaf,
    // otherwise it will be the value after trie smearing.
    // this should be a log value.
    float maxScore;

    public:
        explicit TrieNode(int sym_idx_):symbol_idx(sym_idx_),
                                        maxScore(0.0){}
};

class LexiconTrie
{
    public:
        LexiconTrie();
        ~LexiconTrie()
        {
            if ( m_root_node != nullptr )
            {
                m_root_node = nullptr;
            }
        }
    
    public:
        /* Build the lexicon trie from a lexicon probability file 
            * If give unk_symbol, insert it into nodes without labels.
            */
        int build(std::string lexiconp_file, 
                  std::shared_ptr<asrdec::UnitTable> unit_table, 
                  std::string unk_symbol="", 
                  bool has_prob=true);

        /* Return the root node reference */
        std::shared_ptr<TrieNode> getRoot() { return m_root_node; }

        /* Return the node reference by node_id */
        std::shared_ptr<TrieNode> search(std::vector<int> & idxs);

        /* save trie for debug */
        int save(std::string fname);

        size_t deepth(){ return m_deepth; }

    private:
        /**
         * Smearing the trie using the valid labels inserted in the trie so as to get
         * score on each node (incompleted token).
         * For example, if smear_mode is MAX, then for node "a" in path "c"->"a", we
         * will select the maximum score from all its children like "c"->"a"->"t",
         * "c"->"a"->"n", "c"->"a"->"r"->"e" and so on.
         * This process will be carry out recusively on all the nodes.
         */
        void smear(std::shared_ptr<TrieNode> node, SmearingMode smear_mode);
        double LogAdd(double log_a, double log_b);
        void insert_unk(std::shared_ptr<TrieNode> node, std::string & unk_sym, float & unk_prob);
        void trie_print(std::shared_ptr<TrieNode> node, std::string prefix,  std::vector<std::string> & out_list);
    
    private:
        /* node idx -> node */
        std::shared_ptr<TrieNode> m_root_node;
        std::shared_ptr<asrdec::UnitTable> m_unit_table;
        size_t m_deepth = 0;
};

} // namespace lt
} // namespace asrdec