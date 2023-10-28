/*
 *  Copyright (c) Streamax Tech. and its affiliates.
 *  Authors:
 *         (1) Wang, Yu  Oct 25, 2023
*/
#include <fstream>
#include <iostream>
#include <memory>
#include <cmath>
#include <limits>

#include "lexicon_trie.hpp"

namespace asrdec {
namespace lt
{
    LexiconTrie::LexiconTrie()
    {
        m_root_node = std::make_shared<TrieNode>(-1);
    }

    int LexiconTrie::build(std::string lexiconp_file, 
                           std::shared_ptr<asrdec::UnitTable> unit_table, 
                           std::string unk_symbol, bool has_prob)
    {
        // Read file
        std::ifstream ifile;
        ifile.open( lexiconp_file.c_str() );
        if ( ! ifile.is_open() )
        {
            std::cerr << "Open lexicon file failed! No such file or file is not readable: " << lexiconp_file << std::endl;
            return -1;
        }
        std::string line;
        float unk_prob = 999.0;
        while( getline(ifile, line) )
        {   
            // Skip lines starting with "#" and whitespaces
            if ( line.at(0) == '#' || line.find_first_not_of(" \t\r\n") == std::string::npos ) { continue; }
            /*
             * each line: word log_prob subunits
             * examples: START 0.0 S T A R T
             */

            // std::cout << "Read line: " << line << "\n";

            std::vector<std::string> line_split;
            asrdec::split_string( line, " ", line_split);
            if ( has_prob && line_split.size() < 3)
            {
                ifile.close();
                std::cerr << "each line in lexiconp file should has a format: [word] [log_prob] [subunits]!" << std::endl;
                return -1;
            }
            else if ( (! has_prob) && line_split.size() < 2)
            {
                ifile.close();
                std::cerr << "each line in lexicon file should has a format: [word] [subunits]!" << std::endl;
                return -1;
            }
            // std::cout << line_split.size() << std::endl;

            /*
             * 开始递归创建, 但unk符号会被跳过
             */
            if ( unk_symbol.size() > 0 && line_split[0] == unk_symbol )
            {
                unk_prob = has_prob ? atof(line_split[1].c_str()) : 0.0 ;
                continue;
            }
            std::shared_ptr<TrieNode> parent_node = m_root_node;
            size_t pron_index = has_prob ? 2 : 1;
            // 记录树深度
            m_deepth = std::max( line_split.size() - pron_index, m_deepth );
            // 开始生成
            while ( pron_index < line_split.size() )
            {
                int unit_id = unit_table->idx( line_split[pron_index] );
                
                // std::cout << line_split[pron_index] << "," << unit_id << std::endl;

                /* 查找这个节点是否已经存在 */
                if ( parent_node->children.find( unit_id ) == parent_node->children.end() )
                {
                    parent_node->children[unit_id] = std::make_shared<TrieNode>(unit_id);
                    // parent_node->add_child( unit_id );
                }
                /* 交换父节点 */
                parent_node = parent_node->children[unit_id];

                pron_index ++;
            }
            
            // 将单词与概率存到最后一个输出节点中
            parent_node->labels.push_back( line_split[0] );
            parent_node->scores.push_back( atof(line_split[1].c_str()) );
        }
        ifile.close();

        if ( unk_symbol.size() > 0 )
        {
            /* 在没有输出符号的所有树节点的labels中加入一个UNK标记
            * 使字典树可以在任意一个节点上输出符号
            */

            // 向空白输出节点插入UNK
            if ( has_prob && unk_prob == 999.0 )
            {
                std::cerr << "Build lexicon trie failed! Because unk sumbol is not found in lexicon file!" << std::endl;
                return -1;
            }
            insert_unk( m_root_node, unk_symbol, unk_prob );
        }

        /* 接下来对进行概率前推 */
        if ( has_prob )
        {
            SmearingMode mode = SmearingMode::MAX;
            smear( m_root_node, mode );
        }

        m_unit_table = unit_table;
        return 0;
    }

    std::shared_ptr<TrieNode> LexiconTrie::search(std::vector<int> & idxs)
    {
        std::shared_ptr<TrieNode> search_node = m_root_node;
        for (auto idx : idxs)
        {
            if ( search_node->children.find( idx ) == search_node->children.end() ) 
            {
                return nullptr;
            }
            search_node = search_node->children[ idx ];
        }
        return search_node;
    }

    void LexiconTrie::insert_unk(std::shared_ptr<TrieNode> node, std::string & unk_sym, float & unk_prob)
    {
        if ( node->labels.empty() )
        {
            node->labels.push_back( unk_sym );
            node->scores.push_back( unk_prob );
        }
        if ( ! node->children.empty() )
        {
            for ( auto it = node->children.begin(); it != node->children.end(); it++ )
            {
                insert_unk( it->second, unk_sym, unk_prob);
            }
        }

        return ;
    }

    void LexiconTrie::smear(std::shared_ptr<TrieNode> node, SmearingMode smear_mode)
    {
        node->maxScore = -std::numeric_limits<float>::infinity();
        for ( float score: node->scores )
        {
            if ( smear_mode == SmearingMode::LOGADD )
            {
                node->maxScore = LogAdd(node->maxScore, score);
            }
            else if ( smear_mode == SmearingMode::MAX && 
                      score > node->maxScore )
            {
                node->maxScore = score;
            }
        }

        for ( auto it = node->children.begin(); it != node->children.end(); it++ )
        {
            std::shared_ptr<TrieNode> child_node = it->second;
            smear( child_node, smear_mode );
            if ( smear_mode == SmearingMode::LOGADD )
            {
                node->maxScore = LogAdd(node->maxScore, child_node->maxScore);
            }
            else if ( smear_mode == SmearingMode::MAX && 
                      child_node->maxScore > node->maxScore )
            {
                node->maxScore = child_node->maxScore;
            }
        }
    }

    double LexiconTrie::LogAdd(double log_a, double log_b) 
    {
        double minusdif;
        if (log_a < log_b) 
        {
            std::swap(log_a, log_b);
        }
        minusdif = log_b - log_a;
        if (minusdif < kMinusLogThreshold) 
        {
            return log_a;
        } else 
        {
            return log_a + log1p(exp(minusdif));
        }
    }

    void LexiconTrie::trie_print(std::shared_ptr<TrieNode> node, std::string prefix,  std::vector<std::string> & out_list)
    {
        for ( auto it=node->children.begin(); it!=node->children.end(); it++ )
        {
            std::string line;
            // std::cout << "here3: " << it->first << m_unit_table->sym( it->first ) << std::endl;
            line.append( prefix + m_unit_table->sym( it->first ) );
            
            if ( it->second->labels.size() > 0 )
            {
                line.append( " -> " );
                for ( std::string label : it->second->labels )
                {
                    line.append( label );
                    line.append( " " );
                }
                line.append( " : " );
                line.append( std::to_string(it->second->maxScore) );
            }
            // std::cout << line << std::endl;
            out_list.emplace_back( line );

            trie_print( it->second, prefix + "  ", out_list );
        }

        return ;
    }

    int LexiconTrie::save(std::string fname)
    {
        // 
        std::ofstream ofile;
        ofile.open( fname.c_str() );
        if ( ! ofile.is_open() )
        {
            std::cerr << "Save trie failed! Output file is not accessible: " << fname << std::endl;
            return -1;
        }

        std::vector<std::string> outlist;

        std::shared_ptr<TrieNode> search_node = m_root_node;
        trie_print( search_node, "", outlist );
        // std::cout << "here3: " << outlist.size() << std::endl; 

        for ( std::string line : outlist )
        {
            ofile << line << std::endl;
        }
        ofile.close();

        return 0;
    }

} // namespace lt
} // namespace asedec