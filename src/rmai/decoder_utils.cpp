/*
 *  Copyright (c) Streamax Tech. and its affiliates.
 *  Authors:
 *         (1) Wang, Yu  Oct 25, 2023
*/
#include <fstream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "decoder_utils.hpp"

namespace asrdec
{
const float NEG_INF = -std::numeric_limits<float>::infinity();

float log_add(float probA, float probB)
{
    if ( probA == NEG_INF ) { return probB; }
    else if ( probA == NEG_INF ) { return probA; }
    else
    {
        float max_x = probA > probB ? probA : probB;
        return log( exp( probA - max_x ) + exp( probB - max_x) ) + max_x;
    }
}

float log_add(float probA, float probB, float probC)
{
    return log_add( probA, log_add(probB, probC) );
}

int split_string(const std::string &s, const  std::string seperator, std::vector<std::string> &out_strs)
{	
    out_strs.clear();

    typedef std::string::size_type string_size;

    int swidth = seperator.size();
    string_size i = 0, j = 0;

    while ( (j + swidth) <= s.size() )
    {	
        if ( s.substr(j, swidth) == seperator )
        {   
            if ( j-i > 0 )
            {
                out_strs.push_back( s.substr(i,(j-i)) );
            }
            i = j + swidth;
            j = i;
        }
        else
        {   
            j ++;
        }
    }

    if ( i < s.size() )
    {   
        out_strs.push_back( s.substr(i, s.size()-i) );
    }
    
    return 0;
}

float lm_estimate(std::unique_ptr<lm::ngram::Model> & lm_model,
                    lm::ngram::State & in_state, 
                    std::string & word, 
                    lm::ngram::State & out_state)
{
    lm::WordIndex vocab = lm_model->GetVocabulary().Index( word.c_str() );
    lm::FullScoreReturn ret = lm_model->FullScore(in_state, vocab, out_state);
    return ret.prob;
}

int UnitTable::build(std::string unit_index_file, std::string unk_symbol)
{
    m_syms.clear();
    m_idxs.clear();

    // Read file
    std::ifstream ifile;
    ifile.open( unit_index_file.c_str() );
    if ( ! ifile.is_open() )
    {
        std::cerr << "Open unit_index_file file failed! No such file or file is not readable: " << unit_index_file << std::endl;
        return -1;
    }

    std::string line;
    size_t idx = 0;
    bool found_unk(false);

    while( getline(ifile, line) )
    {   
        // Skip lines starting with "#" and whitespaces
        if ( line.at(0) == '#' || line.find_first_not_of(" \t\r\n") == std::string::npos ) { continue; }
        /*
            * each line: [unit]
            *  examples: START
            */
        std::string::size_type begin = line.find_first_not_of(" \t\r\n");
        std::string unit = line.substr(begin);
        std::string::size_type end = unit.find_first_of(" \t\r\n");
        if ( end != std::string::npos )
        {
            unit = unit.substr(0, end);
        }
        // std::cout << unit << " idx: " << idx << std::endl; 

        if ( unit == unk_symbol ) { found_unk = true; }

        /* 存入,计数 */
        m_syms.push_back( unit );
        m_idxs[ unit ] = idx;

        // std::cout << "here\n"; 
        // break;
        idx ++;
    }
    ifile.close();

    if ( ! found_unk )
    {
        std::cerr << "Build unit table failed! Unk symbol not found in unit file, it is necessary: " << unk_symbol << "\n";
        return -1;
    }

    return 0;
}

void softmax(std::vector<float> &inv, size_t length)
{
    length = ( length == 0 ? inv.size() : std::min(length, inv.size()) );
    float max_value = *max_element(inv.begin(), inv.begin()+length);

    float sum_value = 0.0;
    for (size_t i=0; i<length; i++)
    {
        inv[i] = std::exp( inv[i] - max_value );
        sum_value += inv[i];
    }

    for (size_t i=0; i<length; i++)
    {
        inv[i] /= sum_value;
    }

    return ;
}

int LmPruneTable::build(std::string prune_file, std::shared_ptr<UnitTable> unit_table, size_t max_load_tokens)
{
    std::ifstream ifile;
    ifile.open( prune_file.c_str(), std::ios_base::in );
    if( ! ifile.is_open() )
    {
        std::cerr << "Build lm prune table failed! No such file or file is not readable: " << prune_file << std::endl;
        return -1;
    }

    std::string line;
    std::vector<std::string> tmp;
    int key;
    int idx = 0;
    while ( getline(ifile, line) )
    {   
        split_string(line, "\t", tmp);
        if ( tmp.size() == 1 )
        {
            key = unit_table->idx( tmp[0] );
            if (key == -1) continue;
            m_prune_table[ key ] = std::unordered_map<int, float>();
            idx = 0;
        }
        else
        {
            if ( (max_load_tokens == 0) || (idx < max_load_tokens ) )
            {
                if( tmp.size() != 3 )
                {
                    std::cerr << "Build lm prune table failed! Wrong line format!: " << line << std::endl;
                    return -1;
                }
                m_prune_table[ key ][ unit_table->idx( tmp[1] ) ] = std::atof(tmp[2].c_str());
                idx++;
            }
        }
    }

    /* 概率归一化 */
    for ( auto it=m_prune_table.begin(); it != m_prune_table.end(); it++ )
    {
        std::unordered_map<int,float> & offs = it->second;
        float total = 0.0;
        for ( auto it2=offs.begin(); it2!=offs.end(); it2++)
        {
            total += it2->second;
        }
        for ( auto it2=offs.begin(); it2!=offs.end(); it2++)
        {
            it2->second /= total;
        }
    }

    return 0;
}

std::unordered_map<int,float> & LmPruneTable::get_suf(int pre_id)
{
    return m_prune_table[pre_id];
}

int PruneTable::build(std::string lexicon_file, std::shared_ptr<UnitTable> unit_table)
{
    std::ifstream ifile;
    ifile.open( lexicon_file.c_str(), std::ios_base::in );
    if( ! ifile.is_open() )
    {
        std::cerr << "Build prune table failed! No such file or file is not readable: " << lexicon_file << std::endl;
        return -1;
    }

    std::string line;
    std::vector<std::string> tmp;
    while ( getline(ifile, line) )
    {
        split_string(line, " ", tmp);
        if ( tmp.size() < 2 )
        {
            std::cerr << "Build prune table failed! wrong line format: " << line << std::endl;
            ifile.close();
            return -1;
        }
        // 将头字符加入到头中
        int head_idx = unit_table->idx( tmp[1] );
        if ( find(m_heads.begin(), m_heads.end(), head_idx) == m_heads.end() )
        {
            m_heads.push_back( head_idx );
        }
        // 构建其他上下文
        for ( size_t i=1; i<tmp.size()-1; i++ )
        {
            int prefix_idx = unit_table->idx( tmp[i] );
            int suffix_idx = unit_table->idx( tmp[i+1] );

            if ( m_context.find(prefix_idx) == m_context.end() )
            {
                m_context[ prefix_idx ] = std::vector<int>();
            }
            if ( find(m_context[prefix_idx].begin(), 
                      m_context[prefix_idx].end(), 
                      suffix_idx) == m_context[prefix_idx].end() )
            {
                m_context[prefix_idx].push_back( suffix_idx );
            }
        }
        // 处理尾部
        int tail_idx = unit_table->idx( tmp.back() );
        if ( m_context.find(tail_idx) == m_context.end() )
        {
            m_context[tail_idx] = std::vector<int>();
        }
        if ( find(m_context[tail_idx].begin(), m_context[tail_idx].end(), -1) \
             == m_context[tail_idx].end()
            )
        {
            m_context[tail_idx].push_back( -1 ); // -1 代表下个单次的头部
        }
        
    }
    ifile.close();

    m_unit_table = unit_table;
    return 0;
}

int PruneTable::save(std::string fname)
{
    // 
    std::ofstream ofile;
    ofile.open( fname.c_str() );
    if ( ! ofile.is_open() )
    {
        std::cerr << "Save trie failed! Output file is not accessible: " << fname << std::endl;
        return -1;
    }

    ofile << "[HEAD]\n";
    for ( int & idx: m_heads )
    {
        ofile << m_unit_table->sym( idx ) << "\n";
    }
    ofile << "[CONTEXT]\n";
    for ( auto it = m_context.begin(); it!=m_context.end(); it++ )
    {   
        const int & prefix_idx = it->first;
        std::vector<int> & suffixs = it->second;

        ofile << m_unit_table->sym( prefix_idx ) << "\n";
        for ( int & suffix_idx : suffixs )
        {
            if ( suffix_idx == -1 )
            {
                ofile << "\t<HEAD>\n";
            }
            else
            {
                ofile << "\t" << m_unit_table->sym( suffix_idx ) << "\n";
            }
        }
    }
    ofile.close();

    return 0;
}

// int LookAheadTable::build(std::string lexicon_file, std::string kenlm_file, std::shared_ptr<UnitTable> unit_table)
// {
//     std::ifstream ifile;
//     ifile.open( lexicon_file.c_str(), std::ios_base::in );
//     if( ! ifile.is_open() )
//     {
//         std::cerr << "Build prune table failed! No such file or file is not readable: " << lexicon_file << std::endl;
//         return -1;
//     }

//     std::map<int, std::vector<std::string>> head2word;
//     std::vector<std::string> words;
//     words.emplace_back( "<s>" );

//     std::string line;
//     std::vector<std::string> tmp;
//     while ( getline(ifile, line) )
//     {
//         if ( line.empty() || line[0] == '#' ){ continue; }
//         split_string(line, " ", tmp);

//         std::string & word = tmp[0];
//         int head_idx = unit_table->idx( tmp[1] );

//         if ( find( words.begin(), words.end(), word ) == words.end() )
//         {
//             words.push_back( word );
//         }

//         if ( head2word.find(head_idx) == head2word.end() )
//         {
//             head2word.emplace( head_idx, std::vector<std::string>() );
//         }

//         if ( find( head2word[head_idx].begin(), 
//                    head2word[head_idx].end(), 
//                    word ) == head2word[head_idx].end() )
//         {
//             head2word[head_idx].push_back( word );
//         }
//     }
//     ifile.close();

//     lm::ngram::Config lm_config;
//     kenlm_model = lm::ngram::Model(kenlm_file.c_str(), lm_config );
//     lm::ngram::State dummy_init_state = kenlm_model.NullContextState();
//     lm::ngram::State dummy_in_state, dummy_out_state;

//     for ( std::string & prefix_word : words )
//     {
//         if ( m_context.find(prefix_word) == m_context.end() )
//         {
//             m_context[prefix_word] = std::map<int,std::pair<float,std::string>>();
//         }

//         float score = lm_estimate( kenlm_model, 
//                               dummy_init_state, 
//                                    prefix_word, 
//                                 dummy_in_state );

//         for ( auto it=head2word.begin(); it!=head2word.end(); it++ )
//         {
//             int & head_idx = it->first;
//             std::pair<float,std::string> best_suffix(0,"");
//             for ( std::string & suffix_word : it->second )
//             {
//                 score = lm_estimate( kenlm_model, 
//                                     dummy_in_state,
//                                     suffix_word,
//                                     dummy_out_state
//                                     );
//                 if ( best_suffix.second == "" || score > best_suffix.first )
//                 {
//                     best_suffix.first = score;
//                     best_suffix.second = suffix_word;
//                 }
//             }
//             m_context[ prefix_word ][ head_idx ] = best_score;
//         }
//     }

//     return 0;
// }

// int LookAheadTable::save(std::string fname)
// {
//     std::ofstream ofile;
//     ofile.open( fname.c_str() );
//     if ( ! ofile.is_open() )
//     {
//         std::cerr << "Save look ahead table failed! Output file is not accessible: " << fname << std::endl;
//         return -1;
//     }

//     for ( auto it1=m_context.begin(); it1!=m_context.end(); it1++ )
//     {
//         ofile << it1->first << "\n";
//         for ( auto it2=it1->second.begin(); it2!=it1->second.end(); it2++ )
//         {
//             ofile << "\t" << it2->first;
//             ofile << "\t" << it2->second.first;
//             ofile << "\t" << it2->second.second << "\n";
//         }
//     }
//     ofile.close();

//     return 0;
// }

} // namespace asedec