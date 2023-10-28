#include "lexicon_decoder_wrapper.hpp"
#include "dictionary/Utils.h"
#include "decoder/lm/KenLM.h"
#include "decoder/lm/LM.h"
#include "decoder/Trie.h"

#include <fstream>
#include <iostream>

namespace flashlight
{
    int LexiconDecoderWrapper::init(DecodeParams & dparam)
    {
        /***********************
         * Step 1: 读取子词字典
         ***********************
        */ 
        std::ifstream ifile;
        ifile.open( dparam.unit_dict );
        if ( ! ifile.is_open() )
        {
            std::cerr << "Open file failed! No such file or file is not readable: " << dparam.unit_dict << "\n";
            return -1;
        }
        std::string line;
        int idx = 0;
        while (getline(ifile, line))
        {
            if ( line.size() == 0 || line[0] == '#' )
            {
                continue;
            }
            int j = line.find_first_of(" \t");
            if ( j != std::string::npos ){ line = line.substr(0,j); }

            m_unit_dict[line] = idx;
            m_units.push_back(line);
            idx++;
        }
        ifile.close();
        if ( m_unit_dict.size() == 0 )
        {
            std::cerr << "Read unit dictionary file failed!\n";
            return -1;
        }
        std::cout << "Load units: " << m_unit_dict.size() << "\n";
        for (std::string &w : m_units)
        {
            std::cout << "[" << w << "]\n";
        }

        /***********************
         * Step 2: 读取发音词典,构建前缀树
         ***********************
        */ 
        LexiconMap lexicon = loadWords( dparam.lexicon );
        m_word_dict = createWordDict( lexicon );
        int unk_id = m_word_dict.getIndex( dparam.unk );
        m_lm = std::make_shared<KenLM>( dparam.kenlm_model, m_word_dict );
        m_trie = std::make_shared<Trie>( m_unit_dict.size(), dparam.silence_id );
        LMStatePtr start_state = m_lm->start( false );

        for (auto it=lexicon.begin(); it!=lexicon.end(); it++)
        {
            std::string word = it->first;
            std::vector<std::vector<std::string> > & spellings = it->second;
            int word_id = m_word_dict.getIndex( word );
            std::pair<LMStatePtr, float> lm_score = m_lm->score(start_state, word_id);
            for ( std::vector<std::string> & spelling : spellings )
            {
                std::vector<int> spelling_ids;
                for (std::string & subword : spelling )
                {
                    if ( m_unit_dict.find(subword) == m_unit_dict.end() )
                    {
                        std::cerr << "Lexicon has unknown unit symbol [" << subword << "]\n";
                        return -1;
                    }
                    spelling_ids.emplace_back( m_unit_dict[subword] );
                }
                m_trie->insert( spelling_ids, word_id, lm_score.second );
            }
        }
        m_trie->smear( SmearingMode::MAX ); // SmearingMode.MAX

        /***********************
         * Step 3: 构建decoder backend 
         ***********************
        */ 
        LexiconDecoderOptions doptions;
        doptions.beamSize = dparam.beamSize;
        doptions.beamSizeToken = dparam.beamSizeToken;
        doptions.beamThreshold = dparam.beamThreshold;
        doptions.lmWeight = dparam.lmWeight;
        doptions.logAdd = dparam.logAdd;
        doptions.silScore = dparam.silScore;
        doptions.unkScore = dparam.unkScore;
        doptions.wordScore = dparam.wordScore;

        m_decode_backend.reset( new LexiconDecoder( doptions,
                                                    m_trie,
                                                    m_lm,
                                                    dparam.silence_id,
                                                    dparam.blank_id,
                                                    unk_id,
                                                    m_transitions,
                                                    false
                                                )
                                );

        /***********************
         * Step 4: 其他操作
         ***********************
        */ 
        m_dparam = dparam;
    }

    int LexiconDecoderWrapper::decode(const float *prob_in, size_t T, size_t D, NbestDecodeResult &result_out)
    {
        if ( D != m_units.size() )
        {
            std::cerr << "Probability Dim != Unit classes: " << D << ", " << m_units.size() << "\n";
            return -1;
        }

        std::vector<DecodeResult> raw_result = m_decode_backend->decode(prob_in, (int)T, (int)D);
        result_out.clear();
        int size = raw_result.size() > m_dparam.nbest ? m_dparam.nbest : raw_result.size();
        for ( int i=0; i<size; i++ )
        {
            result_out.units.push_back( std::vector<std::string>() );
            for ( int unit_idx : raw_result[i].tokens )
            {
                result_out.units.back().push_back( m_units[unit_idx] );
            }
            result_out.words.push_back( std::vector<std::string>() );
            for ( int word_idx : raw_result[i].words )
            {
                if ( word_idx >= 0 )
                {

                    result_out.words.back().push_back( m_word_dict.getEntry(word_idx) );
                }
            }   
            result_out.size ++;
        }
        return 0;
    }

    /*Python Interface*/
    #ifdef _PYBIND11
    int LexiconDecoderWrapper::pydecode(const py::array_t<float> &prob_in, NbestDecodeResult &result_out)
    {
        py::buffer_info buffer = prob_in.request();
        if (buffer.ndim != 2)
        {
            std::cerr << "prob_in must a 2-d array\n";
            return -1;
        }
        float *data_ptr = (float *) buffer.ptr;
        return decode(data_ptr, buffer.shape[0], buffer.shape[1], result_out);
    }
    #endif

}

