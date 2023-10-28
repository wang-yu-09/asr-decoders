/*
 *  Copyright (c) Streamax Tech. and its affiliates.
 *  Authors:
 *         (1) Wang, Yu  Oct 25, 2023
*/
#include "prefix_beam_search_decoder.hpp"
#include <math.h>
#include <map>

namespace asrdec {
namespace pbsd
{
    void smooth(std::vector<float> &probin, int factor)
    {   
        float sum_value = 0.0;
        for ( size_t i=0; i<probin.size(); i++ )
        {
            probin[i] = std::cos( 0.5*M_PI*probin[i] )*probin[i];
            sum_value += probin[i];
        }
        for ( size_t i=0; i<probin.size(); i++ )
        {
            sum_value /= sum_value;
        }
        //return softmax(probin);
        return ;
    }

    int PrefixBeamSearchDecoder::init(DecodeParams &dparam)
    {
        /* 构建CTC unit id表 */ 
        m_unit_table = std::make_shared<asrdec::UnitTable>();
        m_unit_table->build( dparam.unit_file );
        
        /* 加载语言模型,本解码器可以不使用语言模型 */
        if ( dparam.unit_lm_weight != 0 && dparam.unit_kenlm_file.size() > 0 )
        {
            lm::ngram::Config lm_config;
            m_unit_kenlm_model.reset( new lm::ngram::Model(dparam.unit_kenlm_file.c_str(), lm_config ) );
            /* 当 前缀为空时，为了使其在和其他 前缀 竞争时具有公平性，使用一个 <unk> 的概率 */
            lm::ngram::State dummy_in_state = m_unit_kenlm_model->NullContextState();
            lm::ngram::State dummy_out_state;
            m_unit_unk_lm_score = lm_estimate( m_unit_kenlm_model, dummy_in_state, dparam.unk_symbol, dummy_out_state );
        }
        else
        {
            m_unit_kenlm_model.reset( nullptr );
            m_unit_unk_lm_score = 0.0;
        }

        /* 保存参数 */
        m_dparam = dparam;

        /* 
            构建词典树
            不插入"unk"符号
        */
        m_lexicon_trie.reset( new asrdec::lt::LexiconTrie() );
        if ( dparam.lexiconp_file.size() > 0 )
        {
            m_lexicon_trie->build( dparam.lexiconp_file, m_unit_table, "", true );
        }
        else if ( dparam.lexicon_file.size() > 0 )
        {
            m_lexicon_trie->build( dparam.lexicon_file, m_unit_table, "", false );
        }
        else
        {
            m_lexicon_trie.reset( nullptr );
        }

        // m_lexicon_trie->save( "/home/wy_rmai/Project/rm_ASR_Decoder/project/danbin/lexitrie.txt" ); 

        /*
            创建词间语言模型
         */
        if ( dparam.wwi_kenlm_file.size() > 0 )
        {
            /* 当前缀为空时，为了使其在和其他 前缀 竞争时具有公平性，使用一个 <unk> 的概率 */
            lm::ngram::Config lm_config;
            m_wwi_kenlm_model.reset( new lm::ngram::Model(dparam.wwi_kenlm_file.c_str(), lm_config ) );
            lm::ngram::State dummy_in_state = m_wwi_kenlm_model->BeginSentenceState();
            lm::ngram::State dummy_out_state; 
            m_wwi_unk_lm_score = lm_estimate( m_wwi_kenlm_model, dummy_in_state, dparam.unk_symbol, dummy_out_state );
        }
        else
        {
            m_wwi_kenlm_model.reset( nullptr );
            m_wwi_unk_lm_score = 0.0;
        }

        /*
          创建单词语言模型
        */
        if ( dparam.word_kenlm_file.size() > 0 )
        {
            /* 当前缀为空时，为了使其在和其他 前缀 竞争时具有公平性，使用一个 <unk> 的概率 */
            lm::ngram::Config lm_config;
            m_word_kenlm_model.reset( new lm::ngram::Model(dparam.word_kenlm_file.c_str(), lm_config ) );
            lm::ngram::State dummy_in_state = m_word_kenlm_model->NullContextState();
            lm::ngram::State dummy_out_state; 
            m_word_unk_lm_score = lm_estimate( m_word_kenlm_model, dummy_in_state, dparam.unk_symbol, dummy_out_state );
        }
        else
        {
            m_word_kenlm_model.reset( nullptr );
            m_word_unk_lm_score = 0.0;
        }

        //std::cout << "m_word_unk_lm_score: " << m_word_unk_lm_score << "\n";

        return 0;
    }

    int PrefixBeamSearchDecoder::init_new_prefix_beam(PrefixBeam &prev_beam, 
                          std::map<std::vector<int>, PrefixBeam> &new_beams, 
                                                std::vector<int> &new_prefix, 
                          std::map<std::vector<int>, PrefixBeam> &pruned_beams,
                                                     const float *prob_in,
                                                          size_t &T, 
                                                          size_t &D,
                                                          size_t &t,
                                                           float &unit_prob,
                                                          size_t &unit_id)
    {
        /* 新beam暂时还不存在当前帧的解码结果中 
         * 如果这个beam存在于上一帧的剪枝结果中, 将它召回
         */
        if ( m_dparam.callback_pruned_beam && pruned_beams.find(new_prefix) != pruned_beams.end() )
        {
            /* 召回后的beam等价于原来的beam的
                * 1. blank分支 = 旧beam的blank分支 + 本帧blank, 旧beam的非blank分支 + 本帧的blank
                * 2. 非blank分支 = 旧beam的非blank + 本帧的非blank
                */
            new_beams[new_prefix] = pruned_beams[new_prefix];
            float log_prob_blank = ( prob_in[ t*D + m_dparam.blank_id ] == 0 ? NEG_INF : log( prob_in[ t*D + m_dparam.blank_id ] ) );
            new_beams[new_prefix].prob_b = asrdec::log_add( pruned_beams[new_prefix].prob_b + log_prob_blank,
                                                            pruned_beams[new_prefix].prob_nb + log_prob_blank);
            new_beams[new_prefix].prob_nb = pruned_beams[new_prefix].prob_nb + unit_prob;        
        }
        /* 否则生成一个新的beam */
        else
        {
            new_beams[new_prefix] = PrefixBeam();
            if ( m_unit_kenlm_model != nullptr )
            {
                new_beams[new_prefix].prob_lm = prev_beam.prob_lm + \
                                                asrdec::lm_estimate(m_unit_kenlm_model, 
                                                                    prev_beam.lm_state,
                                                                    m_unit_table->sym(unit_id),
                                                                    new_beams[new_prefix].lm_state);
            }
        }

        return 0;
    }

    /* decode0: 最基本的前缀束解码算法 */
    int PrefixBeamSearchDecoder::decode0(const float *prob_in, size_t T, size_t D, DecodeResult &result_out)
    {

        if ( D != m_unit_table->size() )
        {
            std::cerr << "The dim != unit table size : " << D << " != " << m_unit_table->size() << "\n";
            return -1;
        }

        std::map<std::vector<int>, PrefixBeam> active_beams;
        std::map<std::vector<int>, PrefixBeam> new_beams;
        std::vector< std::pair<std::vector<int>,PrefixBeam> > new_beams_sorted;
        
        std::vector<size_t> frame_idx;
        for (size_t i=0; i<D; i++){ frame_idx.push_back(i); }

        /* 初始化 一个 active beam */
        std::vector<int> init_prefix;
        active_beams[init_prefix] = PrefixBeam(
                                                0.0, // prob_b
                                                NEG_INF, // prob_nb
                                                0.0, // prob_lm
                                                0.0 // prob_total)
                                            );
        if ( m_unit_kenlm_model != nullptr )
        {
            active_beams[init_prefix].lm_state = m_unit_kenlm_model->NullContextState();
        }

        /* 开始主循环 */
        for ( size_t t = 0; t < T; t ++ )
        {   
            // std::cout << "\n >>>>>>>>>>>>>>>>>>>  frame : " << t << " <<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
            new_beams.clear();
            new_beams_sorted.clear();
            
            /* 排序, 取出TopK个概率值 */
            if ( D > m_dparam.unit_beam_size )
            {
                std::partial_sort(
                                frame_idx.begin(),
                                frame_idx.begin() + m_dparam.unit_beam_size,
                                frame_idx.end(),
                                [&t, &D, &prob_in](const size_t& left, const size_t& right)
                                {
                                    return prob_in[t*D + left] > prob_in[t*D + right];
                                }
                            );
                
            }
            
            // std::cout << "prob top k: " << m_dparam.unit_beam_size << std::endl;
            // for (size_t i=0; i<D; i++)
            // {
            //     std::cout << frame_idx[i] << " " << prob_in.ptr<float>(t)[frame_idx[i]] << "\n";
            // }
            // exit(0);

            // 开始遍历所有的active beam
            for ( auto & prev : active_beams )
            {
                // std::cout << "\nOld prefix: ";
                /* 前缀 */
                const std::vector<int> & prev_prefix = prev.first;
                // for ( const int & idx: prev_prefix )
                // {
                //     std::cout << m_unit_table->sym(idx) << " ";
                // }
                // std::cout << "\n";

                PrefixBeam & prev_beam = prev.second;
                /* 上一个解码的到非blank ID */
                int last_unit_id = prev_prefix.size() > 0 ? prev_prefix.back() : -1 ;
                
                // std::cout << prev_prefix.size() << " " << last_unit_id << std::endl;

                /* 遍历topK个取值 */
                for ( size_t i=0; i<std::min(m_dparam.unit_beam_size,D); i++ )
                {
                    // 得到 unit id 和 log 概率
                    size_t unit_id = frame_idx[i];
                    float unit_prob = prob_in[ t*D + unit_id ];
                    // float unit_prob = topk_prob[i];
                    if ( unit_prob < 0 || unit_prob > 1)
                    {
                        std::cerr << "Decode failed! Prefix beam search decoder need inputs after softmax function but got a value:" << unit_prob << "\n";
                        return -1;
                    }
                    // std::cout << "i=" << i << " unid id=" << unit_id << " unit=" << m_unit_table->sym(unit_id) << " porb=" << unit_prob << std::endl;
                    unit_prob = ( unit_prob == 0 ? NEG_INF : log( unit_prob ) );
                    
                    if ( unit_id == m_dparam.blank_id )
                    {	
                        /* 如果这是一个blank, 将不会产生新的beam
                         * *b + b > *b, *nb + b > *b
                         */
                        if ( new_beams.find( prev_prefix ) == new_beams.end() )
                        {	/* 如果这个beam还没有被本t的解码得到，加入 */
                            new_beams[prev_prefix] = PrefixBeam();
                            // 复用单词序列和语言模型概率
                            if ( m_unit_kenlm_model != nullptr)
                            {
                                new_beams[prev_prefix].lm_state = prev_beam.lm_state;
                                new_beams[prev_prefix].prob_lm = prev_beam.prob_lm;
                            }
                        }
                        /* 更新blank和非blank的概率值 */
                        new_beams[prev_prefix].prob_b = asrdec::log_add(new_beams[prev_prefix].prob_b,
                                                                        prev_beam.prob_b + unit_prob,
                                                                        prev_beam.prob_nb + unit_prob
                                                                    );
                    }
                    else if ( unit_id == last_unit_id )
                    {
                        /* 如果这是一个连续重复的字符,也不会产生新的前缀 
                         * *nb + nb > *nb
                         */
                        if ( new_beams.find( prev_prefix ) == new_beams.end() )
                        {	// 如果这个beam不存在，加入一个
                            new_beams[prev_prefix] = PrefixBeam();
                            // 复用同一个语言模型State和得分
                            if ( m_unit_kenlm_model != nullptr)
                            {
                                new_beams[prev_prefix].lm_state = prev_beam.lm_state;
                                new_beams[prev_prefix].prob_lm = prev_beam.prob_lm;
                            }
                        }
                        /* */
                        new_beams[prev_prefix].prob_nb = asrdec::log_add(new_beams[prev_prefix].prob_nb,
                                                                         prev_beam.prob_nb + unit_prob );

                        /* 如果这是一个重复但不连续(中间被blank隔开)的字符, 会产生新的前缀 
                         * *b + nb > *nb
                         */
                        std::vector<int> new_prefix(prev_prefix.begin(),prev_prefix.end());
                        new_prefix.emplace_back( unit_id );

                        if ( new_beams.find( new_prefix ) == new_beams.end() )
                        {	
                            new_beams[new_prefix] = PrefixBeam();
                            if ( m_unit_kenlm_model != nullptr )
                            {
                                std::string sym = m_unit_table->sym(unit_id);
                                if (new_prefix.size() == 1 || new_prefix.back() == 4)
                                {
                                    sym = std::string("_") + sym;
                                }
                                new_beams[new_prefix].prob_lm = prev_beam.prob_lm + \
                                                                asrdec::lm_estimate(m_unit_kenlm_model, 
                                                                                    prev_beam.lm_state,
                                                                                    sym,
                                                                                    new_beams[new_prefix].lm_state);
                            }
                        }

                        /* 更新blank和非blank的概率值 */
                        new_beams[new_prefix].prob_nb = asrdec::log_add( new_beams[new_prefix].prob_nb, 
                                                                         prev_beam.prob_b + unit_prob );
                    }
                    else
                    {
                        /* 如果这是一个不同的字符,无论如何这将产生新的前缀 
                         * *nb + nb > *nb
                        */
                        std::vector<int> new_prefix(prev_prefix.begin(),prev_prefix.end());
                        new_prefix.emplace_back( unit_id );

                        if ( new_beams.find( new_prefix ) == new_beams.end() )
                        {	
                            new_beams[new_prefix] = PrefixBeam();
                            if ( m_unit_kenlm_model != nullptr )
                            {
                                std::string sym = m_unit_table->sym(unit_id);
                                if (new_prefix.size() == 1 || new_prefix.back() == 4)
                                {
                                    sym = std::string("_") + sym;
                                }
                                new_beams[new_prefix].prob_lm = prev_beam.prob_lm + \
                                                                asrdec::lm_estimate(m_unit_kenlm_model, 
                                                                                    prev_beam.lm_state,
                                                                                    sym,
                                                                                    new_beams[new_prefix].lm_state);
                            }
                        }

                        /* 更新blank和非blank的概率值 */
                        new_beams[new_prefix].prob_nb = asrdec::log_add( new_beams[new_prefix].prob_nb, 
                                                                         prev_beam.prob_b + unit_prob,
                                                                         prev_beam.prob_nb +unit_prob 
                                                                        );
                    }
                }
            }
        
            /* 计算beam的总分并完成剪枝 
             * 语言模型得分将选取得分最高的那个单词序列的得分作为beam的得分
             * 对单词序列进行剪枝
             */
            for ( auto it = new_beams.begin(); it != new_beams.end(); it++ )
            {
                /* 计算总得分 */ 
                it->second.prob_total = log_add(it->second.prob_b, it->second.prob_nb)/(t+1);
                /* 取概率最大值作为语言模型得分(并做长度正则化) 
                 * 如果没有解码出来任何单词,公平起见,给它一个<unk>概率
                 */
                if ( m_unit_kenlm_model != nullptr )
                {   
                    /* 对于空前缀用unk概率 */ 
                    if ( it->first.empty() )
                    {
                        it->second.prob_lm = m_unit_unk_lm_score;
                        it->second.prob_total += m_dparam.unit_lm_weight * it->second.prob_lm;
                    }
                    else
                    {
                        it->second.prob_total += m_dparam.unit_lm_weight * (it->second.prob_lm/it->first.size());
                    }
                }
            }

            new_beams_sorted.clear();
            new_beams_sorted.assign( new_beams.begin(), new_beams.end() );
            /* 排序和剪枝 */
            std::partial_sort( new_beams_sorted.begin(),
                               new_beams_sorted.begin() + std::min( new_beams_sorted.size(), m_dparam.beam_size  ),
                               new_beams_sorted.end(),
                               [](std::pair<std::vector<int>,PrefixBeam> & A, std::pair<std::vector<int>,PrefixBeam> & B)
                               { return A.second.prob_total > B.second.prob_total; }
                            );
            if ( t != T-1 )
            {
                active_beams.clear();
                for ( size_t i=0; i<new_beams_sorted.size(); i++ )
                {
                    if ( i < m_dparam.beam_size )
                    {
                        active_beams[ new_beams_sorted[i].first ] = new_beams_sorted[i].second;
                    }
                }
            }
        }

        /* 保存最后的结果 */
        std::vector<int> & best_prefix = new_beams_sorted.front().first;
        PrefixBeam & best_beam = new_beams_sorted.front().second;

        result_out.unit_ids.swap( best_prefix );
        result_out.units.clear();
        for ( int idx : result_out.unit_ids )
        {
            result_out.units.push_back( m_unit_table->sym( idx ) );
        }
        result_out.am_score = log_add( best_beam.prob_b, best_beam.prob_nb );
        result_out.lm_score = best_beam.prob_lm;

        return 0;
    }

    /* decode1: 0 + 支持跳帧, 热词, 剪枝召回 */
    int PrefixBeamSearchDecoder::decode1(const float *prob_in, size_t T, size_t D, DecodeResult &result_out)
    {
        if ( D != m_unit_table->size() )
        {
            std::cerr << "The dim != unit table size : " << D << " != " << m_unit_table->size() << "\n";
            return -1;
        }

        std::map<std::vector<int>, PrefixBeam> active_beams;
        std::map<std::vector<int>, PrefixBeam> new_beams;
        std::map<std::vector<int>, PrefixBeam> pruned_beams;
        std::vector< std::pair<std::vector<int>,PrefixBeam> > new_beams_sorted;
        
        std::vector<size_t> frame_idx;
        for (size_t i=0; i<D; i++){ frame_idx.push_back(i); }

        /* 初始化 一个 active beam */
        std::vector<int> init_prefix;
        active_beams[init_prefix] = PrefixBeam(
                                                0.0, // prob_b
                                                NEG_INF, // prob_nb
                                                0.0, // prob_lm
                                                0.0 // prob_total)
                                            );
        if ( m_unit_kenlm_model != nullptr )
        {
            active_beams[init_prefix].lm_state = m_unit_kenlm_model->NullContextState();
        }

        /* 开始主循环 */
        size_t continue_silence_count = 0;
        float last_blank_prob = 0.0;
        size_t skipped_frames = 0;
        size_t topk_size = std::min( m_dparam.unit_beam_size, D );

        for ( size_t t = 0; t < T; t ++ )
        {   
            // std::cout << "\n >>>>>>>>>>>>>>>>>>>  frame : " << t << " <<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
            
            new_beams.clear();
            
            /* 声学概率, 取出TopK个概率值 */
            if ( D > m_dparam.unit_beam_size )
            {
                std::partial_sort(
                                frame_idx.begin(),
                                frame_idx.begin() + m_dparam.unit_beam_size,
                                frame_idx.end(),
                                [&t, &D, &prob_in](const size_t& left, const size_t& right)
                                {
                                    return prob_in[t*D + left] > prob_in[t*D + right];
                                }
                            );
            }
            /*
             策略一: 跳帧
             */
            if ( m_dparam.max_silence_frames > 0 )
            {
                if ( find( frame_idx.begin(), frame_idx.begin()+topk_size, m_dparam.blank_id ) \
                     != frame_idx.begin() + topk_size )
                {
                    float blank_prob = prob_in[ t*D+m_dparam.blank_id ];
                    if ( blank_prob > 0.99 ||
                        ( std::abs( blank_prob - last_blank_prob ) <= 0.0001 ) )
                    {
                        continue_silence_count +=1;
                        last_blank_prob = blank_prob;
                        if ( continue_silence_count >= m_dparam.max_silence_frames )
                        {
                            //std::cout << "Skip redundant frame !\n";
                            skipped_frames ++;
                            // 重置和清空一些状态
                            if ( continue_silence_count == m_dparam.max_silence_frames )
                            {
                                pruned_beams.clear();
                            }
                            continue;
                        }
                        else
                        {
                            //std::cout << "Redundant frame counter: " << continue_silence_count << "\n";
                        }
                    }
                    else
                    {
                        //std::cout << "Valid frame, Reset flags 1!\n";
                        continue_silence_count = 0;
                        last_blank_prob = blank_prob;
                    }
                }
                else
                {
                    // std::cout << "Valid frame, Reset flags 2!\n";
                    continue_silence_count = 0;
                    last_blank_prob = 0.0;
                }
            }
            // exit(0);

            // 开始遍历所有的active beam
            for ( auto & prev : active_beams )
            {
                // std::cout << "\nOld prefix: ";
                /* 前缀 */
                const std::vector<int> & prev_prefix = prev.first;
                // for ( const int & idx: prev_prefix )
                // {
                //     std::cout << m_unit_table->sym(idx) << " ";
                // }
                // std::cout << "\n";

                PrefixBeam & prev_beam = prev.second;
                /* 上一个解码的到非blank ID */
                int last_unit_id = prev_prefix.size() > 0 ? prev_prefix.back() : -1 ;

                /* 遍历topK个取值 */
                for ( size_t i=0; i<topk_size; i++)
                {
                    // 得到 unit id 和 log 概率
                    size_t unit_id = frame_idx[i];
                    float unit_prob = prob_in[ t*D + unit_id ];
                    // float unit_prob = topk_prob[i];
                    if ( unit_prob < 0 || unit_prob > 1)
                    {
                        std::cerr << "Decode failed! Prefix beam search decoder need inputs after softmax function but got a value:" << unit_prob << "\n";
                        return -1;
                    }
                    // std::cout << "unid id=" << unit_id << " unit=" << m_unit_table->sym(unit_id) << " porb=" << unit_prob << std::endl;
                    unit_prob = ( unit_prob == 0 ? NEG_INF : log( unit_prob ) );
                    
                    if ( unit_id == m_dparam.blank_id )
                    {	
                        /* 如果这是一个blank, 将不会产生新的beam
                         * *b + b > *b, *nb + b > *b
                         */
                        if ( new_beams.find( prev_prefix ) == new_beams.end() )
                        {	/* 如果这个beam还没有被本t的解码得到，加入 */
                            new_beams[prev_prefix] = PrefixBeam();
                            // 复用单词序列和语言模型概率
                            if ( m_unit_kenlm_model != nullptr )
                            {
                                new_beams[prev_prefix].lm_state = prev_beam.lm_state;
                                new_beams[prev_prefix].prob_lm = prev_beam.prob_lm;
                            }
                        }
                        /* 更新blank和非blank的概率值 */
                        new_beams[prev_prefix].prob_b = asrdec::log_add(new_beams[prev_prefix].prob_b,
                                                                        prev_beam.prob_b + unit_prob,
                                                                        prev_beam.prob_nb + unit_prob
                                                                    );
                    }
                    else if ( unit_id == last_unit_id )
                    {
                        /* 如果这是一个连续重复的字符,也不会产生新的前缀 
                         * *nb + nb > *nb
                         */
                        if ( new_beams.find( prev_prefix ) == new_beams.end() )
                        {	// 如果这个beam不存在，加入一个
                            new_beams[prev_prefix] = PrefixBeam();
                            // 复用同一个语言模型State和得分
                            if ( m_unit_kenlm_model != nullptr)
                            {
                                new_beams[prev_prefix].lm_state = prev_beam.lm_state;
                                new_beams[prev_prefix].prob_lm = prev_beam.prob_lm;
                            }
                        }
                        /* */
                        new_beams[prev_prefix].prob_nb = asrdec::log_add(new_beams[prev_prefix].prob_nb,
                                                                         prev_beam.prob_nb + unit_prob );

                        /* 如果这是一个重复但不连续(中间被blank隔开)的字符, 会产生新的前缀 
                         * *b + nb > *nb
                         */
                        std::vector<int> new_prefix(prev_prefix.begin(),prev_prefix.end());
                        new_prefix.emplace_back( unit_id );

                        if ( new_beams.find( new_prefix ) == new_beams.end() )
                        {
                            init_new_prefix_beam( prev_beam, new_beams, new_prefix, pruned_beams,
                                                  prob_in, T, D, t, unit_prob, unit_id);
                        }

                        /* 更新blank和非blank的概率值 */
                        new_beams[new_prefix].prob_nb = asrdec::log_add( new_beams[new_prefix].prob_nb, 
                                                                         prev_beam.prob_b + unit_prob );
                    }
                    else
                    {
                        /* 如果这是一个不同的字符,无论如何这将产生新的前缀 
                         * *nb + nb > *nb
                        */
                        std::vector<int> new_prefix(prev_prefix.begin(),prev_prefix.end());
                        new_prefix.emplace_back( unit_id );

                        if ( new_beams.find( new_prefix ) == new_beams.end() )
                        {
                            init_new_prefix_beam( prev_beam, new_beams, new_prefix, pruned_beams,
                                                  prob_in, T, D, t, unit_prob, unit_id);
                        }

                        /* 更新blank和非blank的概率值 */
                        new_beams[new_prefix].prob_nb = asrdec::log_add( new_beams[new_prefix].prob_nb, 
                                                                         prev_beam.prob_b + unit_prob,
                                                                         prev_beam.prob_nb +unit_prob 
                                                                        );
                    }
                }
            }
        
            /* 计算beam的总分并完成剪枝 
             * 语言模型得分将选取得分最高的那个单词序列的得分作为beam的得分
             * 对单词序列进行剪枝
             */
            for ( auto it = new_beams.begin(); it != new_beams.end(); it++ )
            {
                /* 计算总得分 */ 
                it->second.prob_total = log_add(it->second.prob_b, it->second.prob_nb)/(t+1-skipped_frames);
                /* 取概率最大值作为语言模型得分(并做长度正则化) 
                 * 如果没有解码出来任何单词,公平起见,给它一个<unk>概率
                 */
                if ( m_unit_kenlm_model != nullptr )
                {   
                    /* 对于空前缀用unk概率 */ 
                    if ( it->first.empty() )
                    {
                        it->second.prob_lm = m_unit_unk_lm_score;
                        it->second.prob_total += m_dparam.unit_lm_weight * it->second.prob_lm;
                    }
                    else
                    {
                        it->second.prob_total += m_dparam.unit_lm_weight * (it->second.prob_lm/it->first.size());
                    }
                }
            }

            // std::cout << "\n";

            new_beams_sorted.clear();
            new_beams_sorted.assign( new_beams.begin(), new_beams.end() );
            /* 排序和剪枝 */
            std::partial_sort( new_beams_sorted.begin(),
                               new_beams_sorted.begin() + std::min( new_beams_sorted.size(), m_dparam.beam_size  ),
                               new_beams_sorted.end(),
                               [](std::pair<std::vector<int>,PrefixBeam> & A, std::pair<std::vector<int>,PrefixBeam> & B)
                               { return A.second.prob_total > B.second.prob_total; }
                            );
            if ( t != T-1 )
            {
                active_beams.clear();
                pruned_beams.clear();
                for ( size_t i=0; i<new_beams_sorted.size(); i++ )
                {   
                    // std::cout << "Prefix: ";
                    // for ( int & uidx : new_beams_sorted[i].first )
                    // {
                    //     std::cout << m_unit_table->sym( uidx ) << " ";
                    // }
                    // std::cout << " Total: " << new_beams_sorted[i].second.prob_total ;
                    // std::cout << " HW: " << ( new_beams_sorted[i].second.prob_hw + new_beams_sorted[i].second.hw_state.score ) ;
                    // std::cout << std::endl;

                    if ( i < m_dparam.beam_size )
                    {
                        active_beams[ new_beams_sorted[i].first ] = new_beams_sorted[i].second;
                    }
                    else if ( m_dparam.callback_pruned_beam ) 
                    {
                        pruned_beams[ new_beams_sorted[i].first ] = new_beams_sorted[i].second;
                    }
                }
            }
        }

        /* 保存最后的结果 */
        std::vector<int> & best_prefix = new_beams_sorted.front().first;
        PrefixBeam & best_beam = new_beams_sorted.front().second;

        // std::cout << "best_prefix size: " << best_prefix.size() << "\n"; 
        // for ( int idx : best_prefix )
        // {
        //     std::cout << idx << " ";
        // }
        // std::cout << "\n";

        result_out.unit_ids.swap( best_prefix );
        result_out.units.clear();
        for ( int idx : result_out.unit_ids )
        {
            result_out.units.push_back( m_unit_table->sym( idx ) );
        }
        result_out.am_score = log_add( best_beam.prob_b, best_beam.prob_nb );
        result_out.lm_score = best_beam.prob_lm;

        return 0;
    }

    /* decode2: 1 + 词典树剪枝 */
    int PrefixBeamSearchDecoder::decode2(const float *prob_in, size_t T, size_t D, DecodeResult &result_out)
    {
        if ( D != m_unit_table->size() )
        {
            std::cerr << "The dim != unit table size : " << D << " != " << m_unit_table->size() << "\n";
            return -1;
        }

        std::vector< std::pair< std::vector<int>, PrefixBeam> > active_beams;
        std::map<std::vector<int>, PrefixBeam> new_beams;
        std::map<std::vector<int>, PrefixBeam> pruned_beams;
        std::vector< std::pair<std::vector<int>,PrefixBeam> > new_beams_sorted;

        /* 初始化 一个 active beam */
        std::vector<int> init_prefix;
        active_beams.emplace_back( init_prefix, 
                                   PrefixBeam(
                                                0.0, // prob_b
                                                NEG_INF, // prob_nb
                                                0.0, // prob_lm
                                                0.0 // prob_total)
                                            ) );
        PrefixBeam & init_beam = active_beams.back().second;
        if ( m_unit_kenlm_model != nullptr )
        {
            init_beam.lm_state = m_unit_kenlm_model->NullContextState();
        }
        // 定义词典树指针
        init_beam.lexicon_trie_nodes.push_back( m_lexicon_trie->getRoot() );

        std::vector<size_t> frame_idx;
        for (size_t i=0; i<D; i++){ frame_idx.push_back(i); }
        
        // topk中存放 idx -> 跳转后得到的词典树指针
        std::map< int, std::vector< std::shared_ptr<asrdec::lt::TrieNode> > > topk;
        std::vector< std::pair<int,float> > topk_tmp;

        /* 开始主循环 */
        size_t continue_silence_count = 0;
        float last_blank_prob = 0.0;
        size_t skipped_frames = 0;

        for ( size_t t = 0; t < T; t ++ )
        {   
            // std::cout << "\n >>>>>>>>>>>>>>>>>>>  frame : " << t << " <<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
            
            new_beams.clear();

            /*********************************
             * 策略一: 跳帧
             * 1. 如果blank的概率持续为0.99的概率超过一定时长, 认为接下来为静音
             * 2. 如果blank的概率很小但收敛, 也认为这段时间为静音
             *********************************/
            if ( m_dparam.max_silence_frames > 0 )
            {
                float blank_prob = prob_in[ t*D + m_dparam.blank_id ];
                if ( blank_prob > 0.0001 )
                {
                    if ( blank_prob > 0.99 || 
                        ( std::abs( blank_prob - last_blank_prob ) <= 0.0001 ) )
                    {
                        continue_silence_count +=1;
                        last_blank_prob = blank_prob;
                        if ( continue_silence_count >= m_dparam.max_silence_frames )
                        {
                            // std::cout << "Skip redundant frame: " << continue_silence_count << "\n";
                            skipped_frames ++;

                            if ( m_dparam.truncate_silence && continue_silence_count == m_dparam.max_silence_frames )
                            {   
                                /* 截断解码beam 
                                   只保留得分最高的那个prefix
                                */
                                std::vector<int> & prev_prefix = active_beams.front().first;
                                std::vector<int> new_prefix(prev_prefix.begin(), prev_prefix.end());
                                active_beams.clear();

                                // new_prefix.push_back( m_dparam.silence_id );
                                active_beams.emplace_back( new_prefix,
                                                           PrefixBeam(
                                                                        0.0, // prob_b
                                                                        NEG_INF, // prob_nb
                                                                        0.0, // prob_lm
                                                                        0.0 // prob_total)
                                                                    ) );
                                if ( m_unit_kenlm_model != nullptr )
                                {
                                    active_beams.back().second.lm_state = m_unit_kenlm_model->NullContextState();
                                }
                                
                                active_beams.back().second.lexicon_trie_nodes.push_back( m_lexicon_trie->getRoot() );

                                if ( active_beams.empty() )
                                {
                                    // std::cerr << "No beam survived after prune!\n";
                                    return -1;
                                }

                                pruned_beams.clear();
                            }
                            continue;
                        }
                        else
                        {
                            // std::cout << "Redundant frame counter: " << continue_silence_count << "\n";
                        }
                    }
                    else
                    {
                        // std::cout << "Valid frame, Reset flags 1!\n";
                        continue_silence_count = 0;
                        last_blank_prob = blank_prob;
                    }
                }
                else
                {
                    // std::cout << "Valid frame, Reset flags 2!\n";
                    continue_silence_count = 0;
                    last_blank_prob = 0.0;
                }
            }

            for ( auto & prev : active_beams )
            {
                /* 前缀 */
                const std::vector<int> & prev_prefix = prev.first;
                // std::cout << "\nOld prefix: ";
                // for ( const int & idx: prev_prefix )
                // {
                //     std::cout << m_unit_table->sym(idx) << " ";
                // }
                // std::cout << "\n";

                PrefixBeam & prev_beam = prev.second;
                /* 上一个解码的到非blank ID */
                int last_unit_id = prev_prefix.size() > 0 ? prev_prefix.back() : -1 ;

                /*********************************
                 * 策略二: 根据前缀,选择接下来可跳转的后缀,然后完成剪枝
                 * 1. blank和与上一帧重复的符号将被保留
                 * 2. 前缀的后缀将会被保留
                 * 3. 如果到达输出节点, 根节点的后缀将会被保留
                 * 因为无需记录历史,所以所有相同的词典树指针只需记录一次即可
                 *********************************/
                topk.clear();
                topk_tmp.clear();
                bool append_root_suffix = false;
                float topk_min_prob = 0.0;

                /* 第一种情况, 可以继续在词典树上向前转播, 但不产生新的单词 */
                for (std::shared_ptr<asrdec::lt::TrieNode> lt_node : prev_beam.lexicon_trie_nodes )
                {   
                    // std::cout << " add for ward suffix: ";
                    for ( auto it=lt_node->children.begin(); it!=lt_node->children.end(); it++ )
                    {   
                        if ( topk.find( it->first ) == topk.end() )
                        {
                            if ( topk_tmp.size() < m_dparam.unit_beam_size || 
                                 prob_in[ t*D + it->first ] > topk_min_prob )
                            {   
                                topk_tmp.emplace_back( it->first, prob_in[ t*D + it->first ] );

                                // std::cout << m_unit_table->sym(it->first) << " ";

                                topk[ it->first ] = std::vector<std::shared_ptr<asrdec::lt::TrieNode>>();
                                topk[ it->first ].push_back( it->second );

                                if ( topk_tmp.size() < m_dparam.unit_beam_size )
                                {
                                    topk_min_prob = std::min( prob_in[ t*D + it->first ], topk_min_prob );
                                }
                            }
                        }
                        else if ( find(topk[it->first].begin(), topk[it->first].end(), it->second ) 
                                  == topk[it->first].end() )
                        {
                            topk[ it->first ].push_back( it->second );
                        }
                    }
                    if ( ! lt_node->labels.empty() ){ append_root_suffix = true; }
                    // std::cout << "\n after add forward subfixs: " << topk.size() << "\n";
                }
                
                /* 第二种情况, 不在词典树上向前传播, 而是根据输出标签, 产生新的单词 */
                if ( append_root_suffix )
                { 
                    for ( auto it=m_lexicon_trie->getRoot()->children.begin(); 
                               it!=m_lexicon_trie->getRoot()->children.end(); it++ )
                    {   
                        if ( topk.find( it->first ) == topk.end() )
                        {
                            if ( topk_tmp.size() < m_dparam.unit_beam_size || 
                                 prob_in[ t*D + it->first ] > topk_min_prob )
                            {   
                                topk_tmp.emplace_back( it->first, prob_in[ t*D + it->first ] );

                                topk[ it->first ] = std::vector<std::shared_ptr<asrdec::lt::TrieNode>>();
                                topk[ it->first ].push_back( it->second );

                                if ( topk_tmp.size() < m_dparam.unit_beam_size )
                                {
                                    topk_min_prob = std::min( prob_in[ t*D + it->first ], topk_min_prob );
                                }
                            }
                        }
                        else if ( find(topk[it->first].begin(), topk[it->first].end(), it->second) 
                                    == topk[it->first].end() )
                        {
                            topk[ it->first ].push_back( it->second );
                        }
                    }
                    // std::cout << " after add root subfixs: " << topk.size() << "\n";
                }

                /* 第三种情况, 加入blank */
                if ( topk_tmp.size() < m_dparam.unit_beam_size || 
                     prob_in[ t*D + m_dparam.blank_id ] > topk_min_prob )
                {
                    topk[ m_dparam.blank_id ] = std::vector<std::shared_ptr<asrdec::lt::TrieNode>>();
                    topk_tmp.emplace_back( m_dparam.blank_id, prob_in[ t*D + m_dparam.blank_id ] );
                    if ( topk_tmp.size() < m_dparam.unit_beam_size )
                    {
                        topk_min_prob = std::min(prob_in[ t*D + m_dparam.blank_id ], topk_min_prob);
                    }
                    // std::cout << " after add blank: " << topk.size() << "\n";
                }

                /* 第四种情况, 加入和前缀相同, 但不会产生新路劲的字符 */
                if ( last_unit_id != -1 && topk.find( last_unit_id ) == topk.end() )
                {
                    if ( topk_tmp.size() < m_dparam.unit_beam_size || 
                         prob_in[ t*D + last_unit_id ] > topk_min_prob )
                    {
                        topk[ last_unit_id ] = std::vector<std::shared_ptr<asrdec::lt::TrieNode>>();
                        topk_tmp.emplace_back( last_unit_id, prob_in[ t*D + last_unit_id ] );
                        
                        // std::cout << " after add same unit: " << topk.size() << "\n";
                    }
                }

                if ( topk_tmp.empty() )
                {
                    std::cerr << "Decode failed! Empty topk?\n";
                    return -1;
                }

                // 对所有候选路劲排序
                size_t topk_size = std::min( topk_tmp.size(), m_dparam.unit_beam_size );
                std::partial_sort(
                                    topk_tmp.begin(),
                                    topk_tmp.begin() + topk_size,
                                    topk_tmp.end(),
                                    []( std::pair<int,float> &A, std::pair<int,float> &B )
                                    { return A.second > B.second; }
                                 );

                /*********************************
                 * 所有候选路径加入结束后, 开始与前缀进行合并
                 *********************************/

                /* 遍历topK个取值 */
                for ( size_t i=0; i<topk_size; i++ )
                {
                    // 得到 unit id 和 log 概率
                    size_t unit_id = topk_tmp[i].first;
                    float unit_prob = topk_tmp[i].second;
                    std::vector<std::shared_ptr<asrdec::lt::TrieNode>> & suffix_ptrs = topk[unit_id];
                    // float unit_prob = topk_prob[i];
                    if ( unit_prob < 0 || unit_prob > 1)
                    {
                        std::cerr << "Decode failed! Prefix beam search decoder need inputs after softmax function but got a value:" << unit_prob << "\n";
                        return -1;
                    }

                    // std::cout << "unid id=" << unit_id << " unit=" << m_unit_table->sym(unit_id) << " porb=" << unit_prob << std::endl;
                    
                    unit_prob = ( unit_prob == 0 ? NEG_INF : log( unit_prob ) );

                    if ( unit_id == m_dparam.blank_id )
                    {	
                        /* 如果这是一个blank, 将不会产生新的beam
                         * *b + b > *b, *nb + b > *b
                         */
                        if ( new_beams.find( prev_prefix ) == new_beams.end() )
                        {	/* 如果这个beam还没有被本t的解码得到，加入*/
                            new_beams[prev_prefix] = PrefixBeam();
                            if ( m_unit_kenlm_model != nullptr )
                            {
                                new_beams[prev_prefix].lm_state = prev_beam.lm_state;
                                new_beams[prev_prefix].prob_lm = prev_beam.prob_lm;
                            }
                            // 没有产生词典树迁移,所以复用之前的
                            new_beams[prev_prefix].lexicon_trie_nodes = prev_beam.lexicon_trie_nodes;
                        }
                        /* 更新blank和非blank的概率值 */
                        new_beams[prev_prefix].prob_b = asrdec::log_add(new_beams[prev_prefix].prob_b,
                                                                        prev_beam.prob_b + unit_prob,
                                                                        prev_beam.prob_nb + unit_prob
                                                                    );
                    }
                    else if ( unit_id == last_unit_id )
                    {
                        /* 如果这是一个连续重复的字符,也不会产生新的前缀
                         * *nb + nb > *nb
                         */
                        if ( new_beams.find( prev_prefix ) == new_beams.end() )
                        {	// 如果这个beam不存在，加入一个
                            new_beams[prev_prefix] = PrefixBeam();
                            if ( m_unit_kenlm_model != nullptr )
                            {
                                new_beams[prev_prefix].lm_state = prev_beam.lm_state;
                                new_beams[prev_prefix].prob_lm = prev_beam.prob_lm;
                            }
                            new_beams[prev_prefix].lexicon_trie_nodes = prev_beam.lexicon_trie_nodes;
                        }
                        /* */
                        new_beams[prev_prefix].prob_nb = asrdec::log_add(new_beams[prev_prefix].prob_nb,
                                                                         prev_beam.prob_nb + unit_prob );

                        /* 如果这是一个重复但不连续(中间被blank隔开)的字符, 会产生新的前缀 
                         * *b + nb > *nb
                         */
                        // 并且如果后缀存在
                        if ( ! suffix_ptrs.empty() )
                        {
                            std::vector<int> new_prefix(prev_prefix.begin(),prev_prefix.end());
                            // if ( unit_id != m_dparam.silence_id )
                            // {
                            new_prefix.emplace_back( unit_id );
                            // }

                            if ( new_beams.find( new_prefix ) == new_beams.end() )
                            {
                                init_new_prefix_beam( prev_beam, new_beams, new_prefix, pruned_beams,
                                                      prob_in, T, D, t, unit_prob, unit_id);
                                // 更新word状态
                                new_beams[ new_prefix ].lexicon_trie_nodes = suffix_ptrs;
                            }

                            /* 更新blank和非blank的概率值 */
                            new_beams[new_prefix].prob_nb = asrdec::log_add( new_beams[new_prefix].prob_nb, 
                                                                             prev_beam.prob_b + unit_prob );
                        }
                    }
                    else
                    {
                        /* 如果这是一个不同的字符,无论如何这将产生新的前缀 
                         * *nb + nb > *nb
                        */
                        // 并且如果后缀存在
                        if ( ! suffix_ptrs.empty() )
                        {
                            std::vector<int> new_prefix(prev_prefix.begin(),prev_prefix.end());
                            // if ( unit_id != m_dparam.silence_id )
                            // {
                            new_prefix.emplace_back( unit_id );
                            // }

                            if ( new_beams.find( new_prefix ) == new_beams.end() )
                            {
                                init_new_prefix_beam( prev_beam, new_beams, new_prefix, pruned_beams,
                                                      prob_in, T, D, t, unit_prob, unit_id);
                                /* 更新blank和非blank的概率值 */
                                new_beams[ new_prefix ].lexicon_trie_nodes = suffix_ptrs;
                            }
                            // 对于相同的前缀,他们的 lexicon_trie_nodes 中携带的词典树节点 应该都是一样了
                            // 所以不用再加进去了

                            /* 更新blank和非blank的概率值 */
                            new_beams[new_prefix].prob_nb = asrdec::log_add( new_beams[new_prefix].prob_nb, 
                                                                             prev_beam.prob_b + unit_prob,
                                                                             prev_beam.prob_nb +unit_prob 
                                                                            );
                        }
                    }
                }
            }
        
            /* 计算beam的总分并完成剪枝 
             * 语言模型得分将选取得分最高的那个单词序列的得分作为beam的得分
             * 对单词序列进行剪枝
             */
            for ( auto it = new_beams.begin(); it != new_beams.end(); it++ )
            {
                /* 计算总得分 */ 
                it->second.prob_total = log_add(it->second.prob_b, it->second.prob_nb)/(t+1-skipped_frames);
                /* 取概率最大值作为语言模型得分(并做长度正则化) 
                 * 如果没有解码出来任何单词,公平起见,给它一个<unk>概率
                 */
                if ( m_unit_kenlm_model != nullptr )
                {   
                    /* 对于空前缀用unk概率 */ 
                    if ( it->first.empty() )
                    {
                        it->second.prob_lm = m_unit_unk_lm_score;
                        it->second.prob_total += m_dparam.unit_lm_weight * it->second.prob_lm;
                    }
                    else
                    {
                        it->second.prob_total += m_dparam.unit_lm_weight * (it->second.prob_lm/it->first.size());
                    }
                }
                /* 计算热词得分 */
            }

            // std::cout << "\n";

            new_beams_sorted.clear();
            new_beams_sorted.assign( new_beams.begin(), new_beams.end() );
            /* 排序和剪枝 */
            std::partial_sort( new_beams_sorted.begin(),
                               new_beams_sorted.begin() + std::min( new_beams_sorted.size(), m_dparam.beam_size  ),
                               new_beams_sorted.end(),
                               [](std::pair<std::vector<int>,PrefixBeam> & A, std::pair<std::vector<int>,PrefixBeam> & B)
                               { return A.second.prob_total > B.second.prob_total; }
                            );
            if ( t != T-1 )
            {
                active_beams.clear();
                pruned_beams.clear();
                for ( size_t i=0; i<new_beams_sorted.size(); i++ )
                {   
                    // std::cout << "Prefix: ";
                    // for ( int & uidx : new_beams_sorted[i].first )
                    // {
                    //     std::cout << m_unit_table->sym( uidx ) << " ";
                    // }
                    // std::cout << " Total: " << new_beams_sorted[i].second.prob_total ;
                    // std::cout << " HW: " << ( new_beams_sorted[i].second.prob_hw + new_beams_sorted[i].second.hw_state.score ) ;
                    // std::cout << std::endl;

                    if ( i < m_dparam.beam_size )
                    {
                        active_beams.push_back( new_beams_sorted[i] );
                    }
                    else if ( m_dparam.callback_pruned_beam ) 
                    {
                        pruned_beams[ new_beams_sorted[i].first ] = new_beams_sorted[i].second;
                    }
                }
            }
        }

        /* 保存最后的结果 */
        std::vector<int> & best_prefix = new_beams_sorted.front().first;
        PrefixBeam & best_beam = new_beams_sorted.front().second;

        // std::cout << "best_prefix size: " << best_prefix.size() << "\n"; 
        // for ( int idx : best_prefix )
        // {
        //     std::cout << idx << " ";
        // }
        // std::cout << "\n";

        result_out.unit_ids.clear();
        result_out.units.clear();
        for ( int idx : best_prefix )
        {   
            if ( idx != m_dparam.silence_id )
            {
                result_out.unit_ids.push_back( idx );
                result_out.units.push_back( m_unit_table->sym( idx ) );
            }
        }
        result_out.am_score = log_add( best_beam.prob_b, best_beam.prob_nb );
        result_out.lm_score = best_beam.prob_lm;

        return 0;
    }

    /* decode3: 2 + 词首语言跳转 */
    int PrefixBeamSearchDecoder::decode3(const float *prob_in, size_t T, size_t D, DecodeResult &result_out)
    {
        if ( D != m_unit_table->size() )
        {
            std::cerr << "The dim != unit table size : " << D << " != " << m_unit_table->size() << "\n";
            return -1;
        }

        std::vector< std::pair< std::vector<int>, PrefixBeam> > active_beams;
        std::map<std::vector<int>, PrefixBeam> new_beams;
        std::map<std::vector<int>, PrefixBeam> pruned_beams;
        std::vector< std::pair<std::vector<int>,PrefixBeam> > new_beams_sorted;

        /* 初始化 一个 active beam */
        std::vector<int> init_prefix;
        active_beams.emplace_back( init_prefix, 
                                   PrefixBeam(
                                                0.0, // prob_b
                                                NEG_INF, // prob_nb
                                                0.0, // prob_lm
                                                0.0 // prob_total)
                                            ) );
        PrefixBeam & init_beam = active_beams.back().second;
        if ( m_unit_kenlm_model != nullptr )
        {
            init_beam.lm_state = m_unit_kenlm_model->NullContextState();
        }
        // 定义词典树指针
        init_beam.word_states.emplace_back(
                                            m_lexicon_trie->getRoot(), // 词典树节点
                                            0.0, // 词间语言模型累计得分
                                            0, // 解码得到的单词数量
                                            m_wwi_kenlm_model->BeginSentenceState() // 词间语言模型状态
                                        );

        std::vector<size_t> frame_idx;
        for (size_t i=0; i<D; i++){ frame_idx.push_back(i); }
        
        // topk中存放 idx -> 跳转后得到的 word state
        std::map< int, std::vector<WordState> > topk;
        std::vector< std::pair<int,float> > topk_tmp;

        /* 开始主循环 */
        size_t continue_silence_count = 0;
        float last_blank_prob = 0.0;
        size_t skipped_frames = 0;

        for ( size_t t = 0; t < T; t ++ )
        {   
            std::cout << "\n >>>>>>>>>>>>>>>  frame : " << t << " / " << T << " <<<<<<<<<<<<<<<" << std::endl;
            
            new_beams.clear();

            /*********************************
             * 策略一: 跳帧
             * 1. 如果blank的概率持续为0.99的概率超过一定时长, 认为接下来为静音
             * 2. 如果blank的概率很小但收敛, 也认为这段时间为静音
             *********************************/
            if ( m_dparam.max_silence_frames > 0 )
            {
                float blank_prob = prob_in[ t*D + m_dparam.blank_id ];
                if ( blank_prob > 0.0001 )
                {
                    if ( blank_prob > 0.99 || 
                        ( std::abs( blank_prob - last_blank_prob ) <= 0.0001 ) )
                    {
                        continue_silence_count +=1;
                        last_blank_prob = blank_prob;
                        if ( continue_silence_count >= m_dparam.max_silence_frames )
                        {
                            // std::cout << "Skip redundant frame: " << continue_silence_count << "\n";
                            skipped_frames ++;

                            if ( m_dparam.truncate_silence && continue_silence_count == m_dparam.max_silence_frames )
                            {   
                                /* 截断解码beam 
                                   只保留得分最高的那个prefix
                                */
                                std::vector<int> & prev_prefix = active_beams.front().first;
                                std::vector<int> new_prefix(prev_prefix.begin(), prev_prefix.end());
                                active_beams.clear();

                                // new_prefix.push_back( m_dparam.silence_id );
                                active_beams.emplace_back( new_prefix,
                                                           PrefixBeam(
                                                                        0.0, // prob_b
                                                                        NEG_INF, // prob_nb
                                                                        0.0, // prob_lm
                                                                        0.0 // prob_total)
                                                                    ) );
                                if ( m_unit_kenlm_model != nullptr )
                                {
                                    active_beams.back().second.lm_state = m_unit_kenlm_model->NullContextState();
                                }

                                active_beams.back().second.word_states.emplace_back(
                                                                                    m_lexicon_trie->getRoot(),
                                                                                    0.0,
                                                                                    0,
                                                                                    m_wwi_kenlm_model->BeginSentenceState()
                                                                                );
   
                                if ( active_beams.empty() )
                                {
                                    // std::cerr << "No beam survived after prune!\n";
                                    return -1;
                                }

                                pruned_beams.clear();
                            }
                            continue;
                        }
                        else
                        {
                            // std::cout << "Redundant frame counter: " << continue_silence_count << "\n";
                        }
                    }
                    else
                    {
                        // std::cout << "Valid frame, Reset flags 1!\n";
                        continue_silence_count = 0;
                        last_blank_prob = blank_prob;
                    }
                }
                else
                {
                    // std::cout << "Valid frame, Reset flags 2!\n";
                    continue_silence_count = 0;
                    last_blank_prob = 0.0;
                }
            }

            //std::cout << "here 1\n";

            for ( auto & prev : active_beams )
            {
                /* 前缀 */
                const std::vector<int> & prev_prefix = prev.first;
                // std::cout << "\nOld prefix: ";
                // for ( const int & idx: prev_prefix )
                // {
                //     std::cout << m_unit_table->sym(idx) << " ";
                // }
                // std::cout << "\n";

                PrefixBeam & prev_beam = prev.second;
                /* 上一个解码的到非blank ID */
                int last_unit_id = prev_prefix.size() > 0 ? prev_prefix.back() : -1 ;

                /*********************************
                 * 策略二: 根据前缀,选择接下来可跳转的后缀,然后完成剪枝
                 * 1. blank和与上一帧重复的符号将被保留
                 * 2. 前缀的后缀将会被保留
                 * 3. 如果到达输出节点, 根节点的后缀将会被保留
                 *********************************/
                topk.clear();
                topk_tmp.clear();

                std::vector<WordState *> wi_trans;
                float topk_min_score = 0.0;
                
                /* 第一种情况, 可以继续在词典树上向前转播, 但不产生新的单词 */
                for ( WordState & wstate : prev_beam.word_states )
                {   
                    std::shared_ptr<asrdec::lt::TrieNode> lt_node = wstate.lexicon_trie_node;
                    // std::cout << " add for ward suffix: ";

                    for ( auto it=lt_node->children.begin(); it!=lt_node->children.end(); it++ )
                    {   
                        /* 如果这个topk还不存在, 尝试创建 */
                        if ( topk.find( it->first ) == topk.end() )
                        {
                            if ( topk_tmp.size() < m_dparam.unit_beam_size || 
                                 prob_in[ t*D + it->first ] > topk_min_score )
                            {
                                topk[ it->first ] = std::vector<WordState>();
                                topk_tmp.emplace_back( it->first, prob_in[ t*D + it->first ] );

                                // std::cout << m_unit_table->sym(it->first) << " ";

                                topk[ it->first ].emplace_back( 
                                                                it->second,
                                                                wstate.wwi_score,   //继承词间得分
                                                                wstate.n_words,     //继承词间计数
                                                                wstate.wwi_lm_state //继承词间状态
                                                            );
                                // 更新topk最小概率
                                if ( topk_tmp.size() < m_dparam.unit_beam_size )
                                {
                                    topk_min_score = std::min( topk_min_score, prob_in[ t*D + it->first ] );
                                }
                            }
                        }
                        else
                        {
                            topk[ it->first ].emplace_back( 
                                                            it->second,
                                                            wstate.wwi_score, //继承词间得分
                                                            wstate.n_words,  //继承词间计数
                                                            wstate.wwi_lm_state //继承词间状态
                                                        );
                        }
                    }
                    // 如果这个节点可以输出单词, 记录它
                    if ( ! lt_node->labels.empty() )
                    {
                        wi_trans.push_back( &wstate );
                    }
                    //std::cout << "\n after add forward subfixs: " << topk.size() << "\n";
                }

                // size_t c1 = 0;
                // for ( auto it=topk.begin(); it!=topk.end(); it++ )
                // {
                //     c1 += it->second.size();
                // }
                // std::cout << " after add forward subfixs: unit size: " << topk.size() << " , state size: " << c1 << "\n";

                /* 第二种情况, 在可以输出单词的节点上, 产生新的单词 */
                if ( ! wi_trans.empty() )
                {
                    for ( auto it=m_lexicon_trie->getRoot()->children.begin(); 
                               it!=m_lexicon_trie->getRoot()->children.end(); it++ )
                    {   
                        if ( topk.find( it->first ) == topk.end() )
                        {
                            if ( topk_tmp.size() < m_dparam.unit_beam_size || 
                                prob_in[ t*D + it->first ] > topk_min_score )
                            {
                                topk_tmp.emplace_back( it->first, prob_in[ t*D + it->first ] );
                                topk[ it->first ] = std::vector<WordState>();

                                for ( WordState * wstate : wi_trans )
                                {
                                    topk[ it->first ].emplace_back( 
                                                                    it->second,
                                                                    wstate->wwi_score, //先继承词间得分
                                                                    wstate->n_words + 1,  //计数加一
                                                                    wstate->wwi_lm_state //先继承状态
                                                                );
                                    // 重新计算词间语言模型得分
                                    topk[ it->first ].back().wwi_score += lm_estimate( m_wwi_kenlm_model, 
                                                                            wstate->wwi_lm_state, 
                                                                            m_unit_table->sym( it->first ), 
                                                                            topk[ it->first ].back().wwi_lm_state
                                                                            );
                                }
                                // 更新topk最小概率
                                if ( topk_tmp.size() < m_dparam.unit_beam_size )
                                {
                                    topk_min_score = std::min( topk_min_score, prob_in[ t*D + it->first ] );
                                }
                            }
                        }
                        else 
                        {
                            for ( WordState * wstate : wi_trans )
                            {
                                topk[ it->first ].emplace_back( 
                                                                it->second,
                                                                wstate->wwi_score, //先继承词间得分
                                                                wstate->n_words + 1,  //计数加一
                                                                wstate->wwi_lm_state //先继承状态
                                                            );
                                // 重新计算词间语言模型得分
                                topk[ it->first ].back().wwi_score += lm_estimate( m_wwi_kenlm_model, 
                                                                                  wstate->wwi_lm_state, 
                                                                                  m_unit_table->sym( it->first ), 
                                                                                  topk[ it->first ].back().wwi_lm_state
                                                                        );
                            }
                        }
                    }
                    // std::cout << " add root subfixs: " << m_lexicon_trie->getRoot()->children.size() << "\n";
                }

                // c1 = 0;
                // for ( auto it=topk.begin(); it!=topk.end(); it++ )
                // {
                //     c1 += it->second.size();
                // }
                // std::cout << " after add root subfixs: unit size: " << topk.size() << " , state size: " << c1 << "\n";

                /* 第三种情况, 加入blank */
                if ( topk_tmp.size() < m_dparam.unit_beam_size || 
                    prob_in[ t*D + m_dparam.blank_id ] > topk_min_score )
                {
                    topk[ m_dparam.blank_id ] = std::vector<WordState>();
                    topk_tmp.emplace_back( m_dparam.blank_id, prob_in[ t*D + m_dparam.blank_id ] );
                    // std::cout << " add blank: 1\n";
                    if ( topk_tmp.size() < m_dparam.unit_beam_size )
                    {
                        topk_min_score = std::min(topk_min_score, prob_in[ t*D + m_dparam.blank_id ]);
                    }
                }

                /* 第四种情况, 加入和前缀相同, 但不会产生新路径的字符 */
                if ( last_unit_id != -1 && topk.find( last_unit_id ) == topk.end() )
                {
                    if ( topk_tmp.size() < m_dparam.unit_beam_size || 
                         prob_in[ t*D + last_unit_id ] > topk_min_score )
                    {
                        topk[ last_unit_id ] = std::vector<WordState>();
                        topk_tmp.emplace_back( last_unit_id, prob_in[ t*D + last_unit_id ] );
                    }
                    // std::cout << " add same suffix: 1\n";
                }

                // c1 = 0;
                // for ( auto it=topk.begin(); it!=topk.end(); it++ )
                // {
                //     c1 += it->second.size();
                // }
                // std::cout << " after add blank & same subfixs: unit size: " << topk.size() << " , state size: " << c1 << "\n";

                if ( topk_tmp.empty() )
                {
                    std::cerr << "Decode failed! Empty topk?\n";
                    return -1;
                }

                // 对所有候选路径排序
                size_t topk_size = std::min( topk_tmp.size(), m_dparam.unit_beam_size );
                std::partial_sort(
                                    topk_tmp.begin(),
                                    topk_tmp.begin() + topk_size,
                                    topk_tmp.end(),
                                    []( std::pair<int,float> &A, std::pair<int,float> &B )
                                    { return A.second > B.second; }
                                 );

                /*********************************
                 * 开始合并前缀和候选路径
                 *********************************/

                /* 遍历topK个取值 */
                for ( size_t i=0; i<topk_size; i++ )
                {
                    // 得到 unit id 和 log 概率
                    size_t unit_id = topk_tmp[i].first;
                    float unit_prob = topk_tmp[i].second;
                    std::vector<WordState> & suffix_ptrs = topk[unit_id];
                    // float unit_prob = topk_prob[i];
                    if ( unit_prob < 0 || unit_prob > 1 )
                    {
                        std::cerr << "Decode failed! Prefix beam search decoder need inputs after softmax function but got a value:" << unit_prob << "\n";
                        return -1;
                    }

                    // std::cout << "unid id=" << unit_id << " unit=" << m_unit_table->sym(unit_id) << " porb=" << unit_prob << std::endl;
                    
                    unit_prob = ( unit_prob == 0 ? NEG_INF : log( unit_prob ) );

                    if ( unit_id == m_dparam.blank_id )
                    {	
                        /* 如果这是一个blank, 将不会产生新的beam
                         * *b + b > *b, *nb + b > *b
                         */
                        if ( new_beams.find( prev_prefix ) == new_beams.end() )
                        {	/* 如果这个beam还没有被本t的解码得到，加入*/
                            new_beams[prev_prefix] = PrefixBeam();
                            if ( m_unit_kenlm_model != nullptr )
                            {
                                new_beams[prev_prefix].lm_state = prev_beam.lm_state;
                                new_beams[prev_prefix].prob_lm = prev_beam.prob_lm;
                            }
                            new_beams[prev_prefix].word_states = prev_beam.word_states;
                        }
                        /* 更新blank和非blank的概率值 */
                        new_beams[prev_prefix].prob_b = asrdec::log_add(new_beams[prev_prefix].prob_b,
                                                                        prev_beam.prob_b + unit_prob,
                                                                        prev_beam.prob_nb + unit_prob
                                                                    );
                    }
                    else if ( unit_id == last_unit_id )
                    {
                        /* 如果这是一个连续重复的字符,也不会产生新的前缀
                         * *nb + nb > *nb
                         */
                        if ( new_beams.find( prev_prefix ) == new_beams.end() )
                        {	// 如果这个beam不存在，加入一个
                            new_beams[prev_prefix] = PrefixBeam();
                            if ( m_unit_kenlm_model != nullptr )
                            {
                                new_beams[prev_prefix].lm_state = prev_beam.lm_state;
                                new_beams[prev_prefix].prob_lm = prev_beam.prob_lm;
                            }
                            new_beams[prev_prefix].word_states = prev_beam.word_states;
                        }
                        /* */
                        new_beams[prev_prefix].prob_nb = asrdec::log_add(new_beams[prev_prefix].prob_nb,
                                                                         prev_beam.prob_nb + unit_prob );

                        /* 如果这是一个重复但不连续(中间被blank隔开)的字符, 会产生新的前缀 
                         * *b + nb > *nb
                         */
                        // 并且如果后缀存在
                        if ( ! suffix_ptrs.empty() )
                        {
                            std::vector<int> new_prefix(prev_prefix.begin(),prev_prefix.end());
                            // if ( unit_id != m_dparam.silence_id )
                            // {
                            new_prefix.emplace_back( unit_id );
                            // }

                            if ( new_beams.find( new_prefix ) == new_beams.end() )
                            {
                                init_new_prefix_beam( prev_beam, new_beams, new_prefix, pruned_beams,
                                                      prob_in, T, D, t, unit_prob, unit_id);
                                // 更新word状态
                                new_beams[ new_prefix ].word_states = suffix_ptrs;
                            }
                            
                            /* 更新blank和非blank的概率值 */
                            new_beams[new_prefix].prob_nb = asrdec::log_add( new_beams[new_prefix].prob_nb, 
                                                                             prev_beam.prob_b + unit_prob );
                        }
                    }
                    else
                    {
                        /* 如果这是一个不同的字符,无论如何这将产生新的前缀 
                         * *nb + nb > *nb
                        */
                        // 并且如果后缀存在
                        if ( ! suffix_ptrs.empty() )
                        {
                            std::vector<int> new_prefix(prev_prefix.begin(),prev_prefix.end());
                            // if ( unit_id != m_dparam.silence_id )
                            // {
                            new_prefix.emplace_back( unit_id );
                            // }

                            if ( new_beams.find( new_prefix ) == new_beams.end() )
                            {
                                init_new_prefix_beam( prev_beam, new_beams, new_prefix, pruned_beams,
                                                      prob_in, T, D, t, unit_prob, unit_id);
                                /* 更新blank和非blank的概率值 */
                                new_beams[ new_prefix ].word_states = suffix_ptrs;
                            }
                            /* 更新blank和非blank的概率值 */
                            new_beams[new_prefix].prob_nb = asrdec::log_add( new_beams[new_prefix].prob_nb, 
                                                                            prev_beam.prob_b + unit_prob,
                                                                            prev_beam.prob_nb +unit_prob 
                                                                            );
                        }
                    }
                }
            }
            //std::cout << "here 2\n";
            /* 计算beam的总分并完成剪枝 
             * 语言模型得分将选取得分最高的那个单词序列的得分作为beam的得分
             * 对单词序列进行剪枝
             */
            for ( auto it = new_beams.begin(); it != new_beams.end(); it++ )
            {
                /* 计算总得分 */ 
                it->second.prob_total = log_add(it->second.prob_b, it->second.prob_nb)/(t+1-skipped_frames);
                /* 取概率最大值作为语言模型得分(并做长度正则化) 
                 * 如果没有解码出来任何单词,公平起见,给它一个<unk>概率
                 */
                if ( m_unit_kenlm_model != nullptr )
                {   
                    /* 对于空前缀用unk概率 */ 
                    if ( it->first.empty() )
                    {
                        it->second.prob_lm = m_unit_unk_lm_score;
                        it->second.prob_total += m_dparam.unit_lm_weight * it->second.prob_lm;
                    }
                    else
                    {
                        it->second.prob_total += m_dparam.unit_lm_weight * (it->second.prob_lm/it->first.size());
                    }
                }
                /* 计算热词得分 */
                /* 计算词间语言模型得分
                   查找所有词间跳转路径里面得分最高的那个
                 */
                float best_wwi_lm_score = NEG_INF;
                for ( WordState & wstate: it->second.word_states )
                {
                    if ( wstate.n_words == 0 && m_wwi_unk_lm_score > best_wwi_lm_score )
                    {
                        best_wwi_lm_score = m_wwi_unk_lm_score;
                    }
                    else if ( wstate.wwi_score/wstate.n_words > best_wwi_lm_score )
                    {
                        best_wwi_lm_score = wstate.wwi_score/wstate.n_words;
                    }
                }
                it->second.prob_total += m_dparam.wwi_lm_weight * best_wwi_lm_score;
            }
            //std::cout << "here 3\n";
            // std::cout << "\n";

            new_beams_sorted.clear();
            new_beams_sorted.assign( new_beams.begin(), new_beams.end() );
            /* 排序和剪枝 */
            std::partial_sort( new_beams_sorted.begin(),
                               new_beams_sorted.begin() + std::min( new_beams_sorted.size(), m_dparam.beam_size  ),
                               new_beams_sorted.end(),
                               [](std::pair<std::vector<int>,PrefixBeam> & A, std::pair<std::vector<int>,PrefixBeam> & B)
                               { return A.second.prob_total > B.second.prob_total; }
                            );
            if ( t != T-1 )
            {
                active_beams.clear();
                pruned_beams.clear();
                for ( size_t i=0; i<new_beams_sorted.size(); i++ )
                {   
                    // std::cout << "Prefix: ";
                    // for ( int & uidx : new_beams_sorted[i].first )
                    // {
                    //     std::cout << m_unit_table->sym( uidx ) << " ";
                    // }
                    // std::cout << " Total: " << new_beams_sorted[i].second.prob_total ;
                    // std::cout << " HW: " << ( new_beams_sorted[i].second.prob_hw + new_beams_sorted[i].second.hw_state.score ) ;
                    // std::cout << std::endl;

                    if ( i < m_dparam.beam_size )
                    {
                        active_beams.push_back( new_beams_sorted[i] );
                    }
                    else if ( m_dparam.callback_pruned_beam ) 
                    {
                        pruned_beams[ new_beams_sorted[i].first ] = new_beams_sorted[i].second;
                    }
                }
            }
            //std::cout << "here 4\n";
        }

        /* 保存最后的结果 */
        std::vector<int> & best_prefix = new_beams_sorted.front().first;
        PrefixBeam & best_beam = new_beams_sorted.front().second;

        // std::cout << "best_prefix size: " << best_prefix.size() << "\n"; 
        // for ( int idx : best_prefix )
        // {
        //     std::cout << idx << " ";
        // }
        // std::cout << "\n";

        result_out.unit_ids.clear();
        result_out.units.clear();
        for ( int idx : best_prefix )
        {   
            if ( idx != m_dparam.silence_id )
            {
                result_out.unit_ids.push_back( idx );
                result_out.units.push_back( m_unit_table->sym( idx ) );
            }
        }
        result_out.am_score = log_add( best_beam.prob_b, best_beam.prob_nb );
        result_out.lm_score = best_beam.prob_lm;

        return 0;
    }

    /* decode4: 3 + 单词语言模型 */
    int PrefixBeamSearchDecoder::decode4(const float *prob_in, size_t T, size_t D, DecodeResult &result_out)
    {
        if ( D != m_unit_table->size() )
        {
            std::cerr << "The dim != unit table size : " << D << " != " << m_unit_table->size() << "\n";
            return -1;
        }

        std::vector< std::pair< std::vector<int>, PrefixBeam> > active_beams;
        std::map<std::vector<int>, PrefixBeam> new_beams;
        std::map<std::vector<int>, PrefixBeam> pruned_beams;
        std::vector< std::pair<std::vector<int>,PrefixBeam> > new_beams_sorted;

        /* 初始化 一个 active beam */
        std::vector<int> init_prefix;
        active_beams.emplace_back( init_prefix, 
                                   PrefixBeam(
                                                0.0, // prob_b
                                                NEG_INF, // prob_nb
                                                0.0, // prob_lm
                                                0.0 // prob_total)
                                            ) );
        PrefixBeam & init_beam = active_beams.back().second;
        if ( m_unit_kenlm_model != nullptr )
        {
            init_beam.lm_state = m_unit_kenlm_model->NullContextState();
        }
        // 定义词典树指针
        init_beam.word_states.emplace_back(
                                            m_lexicon_trie->getRoot(), // 词典树节点
                                            0.0, // 词间语言模型累计得分
                                            0, // 解码得到的单词数量
                                            m_wwi_kenlm_model->BeginSentenceState(), // 词间语言模型状态
                                            0.0,
                                            m_word_kenlm_model->NullContextState()
                                        );

        std::vector<size_t> frame_idx;
        for (size_t i=0; i<D; i++){ frame_idx.push_back(i); }
        
        // topk中存放 idx -> 跳转后得到的 word state
        std::map< int, std::vector<WordState> > topk;
        std::vector< std::pair<int,float> > topk_tmp;

        /* 开始主循环 */
        size_t continue_silence_count = 0;
        float last_blank_prob = 0.0;
        size_t skipped_frames = 0;

        for ( size_t t = 0; t < T; t ++ )
        {   
            // std::cout << "\n >>>>>>>>>>>>>>>>>>>  frame : " << t << " <<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
            
            new_beams.clear();

            /*********************************
             * 策略一: 跳帧
             * 1. 如果blank的概率持续为0.99的概率超过一定时长, 认为接下来为静音
             * 2. 如果blank的概率很小但收敛, 也认为这段时间为静音
             *********************************/
            if ( m_dparam.max_silence_frames > 0 )
            {
                float blank_prob = prob_in[ t*D + m_dparam.blank_id ];
                if ( blank_prob > 0.0001 )
                {
                    if ( blank_prob > 0.99 || 
                        ( std::abs( blank_prob - last_blank_prob ) <= 0.0001 ) )
                    {
                        continue_silence_count +=1;
                        last_blank_prob = blank_prob;
                        if ( continue_silence_count >= m_dparam.max_silence_frames )
                        {
                            // std::cout << "Skip redundant frame: " << continue_silence_count << "\n";
                            skipped_frames ++;

                            if ( m_dparam.truncate_silence && continue_silence_count == m_dparam.max_silence_frames )
                            {   
                                /* 截断解码beam 
                                   只保留得分最高的那个prefix
                                */
                                std::vector<int> & prev_prefix = active_beams.front().first;
                                std::vector<int> new_prefix(prev_prefix.begin(), prev_prefix.end());
                                active_beams.clear();

                                // new_prefix.push_back( m_dparam.silence_id );
                                active_beams.emplace_back( new_prefix,
                                                           PrefixBeam(
                                                                        0.0, // prob_b
                                                                        NEG_INF, // prob_nb
                                                                        0.0, // prob_lm
                                                                        0.0 // prob_total)
                                                                    ) );
                                if ( m_unit_kenlm_model != nullptr )
                                {
                                    active_beams.back().second.lm_state = m_unit_kenlm_model->NullContextState();
                                }

                                active_beams.back().second.word_states.emplace_back(
                                                                                    m_lexicon_trie->getRoot(),
                                                                                    0.0,
                                                                                    0,
                                                                                    m_wwi_kenlm_model->BeginSentenceState(),
                                                                                    0.0,
                                                                                    m_word_kenlm_model->NullContextState()
                                                                                );
   
                                if ( active_beams.empty() )
                                {
                                    // std::cerr << "No beam survived after prune!\n";
                                    return -1;
                                }

                                pruned_beams.clear();
                            }
                            continue;
                        }
                        else
                        {
                            // std::cout << "Redundant frame counter: " << continue_silence_count << "\n";
                        }
                    }
                    else
                    {
                        // std::cout << "Valid frame, Reset flags 1!\n";
                        continue_silence_count = 0;
                        last_blank_prob = blank_prob;
                    }
                }
                else
                {
                    // std::cout << "Valid frame, Reset flags 2!\n";
                    continue_silence_count = 0;
                    last_blank_prob = 0.0;
                }
            }

            for ( auto & prev : active_beams )
            {
                /* 前缀 */
                const std::vector<int> & prev_prefix = prev.first;
                // std::cout << "\nOld prefix: ";
                // for ( const int & idx: prev_prefix )
                // {
                //     std::cout << m_unit_table->sym(idx) << " ";
                // }
                // std::cout << "\n";

                PrefixBeam & prev_beam = prev.second;
                /* 上一个解码的到非blank ID */
                int last_unit_id = prev_prefix.size() > 0 ? prev_prefix.back() : -1 ;

                /*********************************
                 * 策略二: 根据前缀,选择接下来可跳转的后缀,然后完成剪枝
                 * 1. blank和与上一帧重复的符号将被保留
                 * 2. 前缀的后缀将会被保留
                 * 3. 如果到达输出节点, 根节点的后缀将会被保留
                 *********************************/
                topk.clear();
                topk_tmp.clear();

                std::vector<WordState *> wi_trans;
                float topk_min_score = 0.0;
                
                /* 第一种情况, 可以继续在词典树上向前转播, 但不产生新的单词 */
                for ( WordState & wstate : prev_beam.word_states )
                {   
                    std::shared_ptr<asrdec::lt::TrieNode> lt_node = wstate.lexicon_trie_node;
                    // std::cout << " add for ward suffix: ";

                    for ( auto it=lt_node->children.begin(); it!=lt_node->children.end(); it++ )
                    {   
                        /* 如果这个topk还不存在, 尝试创建 */
                        if ( topk.find( it->first ) == topk.end() )
                        {
                            if ( topk_tmp.size() < m_dparam.unit_beam_size || 
                                 prob_in[ t*D + it->first ] > topk_min_score )
                            {
                                topk[ it->first ] = std::vector<WordState>();
                                topk_tmp.emplace_back( it->first, prob_in[ t*D + it->first ] );

                                // std::cout << m_unit_table->sym(it->first) << " ";

                                topk[ it->first ].emplace_back( 
                                                                it->second,
                                                                wstate.wwi_score,   //继承词间得分
                                                                wstate.n_words,     //继承词间计数
                                                                wstate.wwi_lm_state, //继承词间状态
                                                                wstate.word_score,
                                                                wstate.word_lm_state
                                                            );
                                // 更新topk最小概率
                                if ( topk_tmp.size() < m_dparam.unit_beam_size )
                                {
                                    topk_min_score = std::min( topk_min_score, prob_in[ t*D + it->first ] );
                                }
                            }
                        }
                        else
                        {
                            topk[ it->first ].emplace_back( 
                                                            it->second,
                                                            wstate.wwi_score, //继承词间得分
                                                            wstate.n_words,  //继承词间计数
                                                            wstate.wwi_lm_state, //继承词间状态
                                                            wstate.word_score,
                                                            wstate.word_lm_state
                                                        );
                        }
                    }
                    // 如果这个节点可以输出单词, 记录它, 并更新他的语言模型得分
                    if ( ! lt_node->labels.empty() )
                    {
                        if ( m_word_kenlm_model != nullptr )
                        {
                            /* 更新单词级语言模型得分 */
                            lm::ngram::State best_state, tmp_state;
                            float best_score = NEG_INF;

                            for ( std::string & label: lt_node->labels )
                            {
                                float s = lm_estimate( m_word_kenlm_model, wstate.word_lm_state, label, tmp_state);
                                if ( s > best_score )
                                {
                                    best_score = s;
                                    best_state = tmp_state;
                                }
                            }
                            wstate.word_score = best_score;
                            wstate.word_lm_state = best_state;
                        }
                        wi_trans.push_back( &wstate );
                    }
                    //std::cout << "\n after add forward subfixs: " << topk.size() << "\n";
                }

                // size_t c1 = 0;
                // for ( auto it=topk.begin(); it!=topk.end(); it++ )
                // {
                //     c1 += it->second.size();
                // }
                // std::cout << " after add forward subfixs: unit size: " << topk.size() << " , state size: " << c1 << "\n";

                /* 第二种情况, 在可以输出单词的节点上, 产生新的单词 */
                if ( ! wi_trans.empty() )
                {
                    for ( auto it=m_lexicon_trie->getRoot()->children.begin(); 
                               it!=m_lexicon_trie->getRoot()->children.end(); it++ )
                    {   
                        if ( topk.find( it->first ) == topk.end() )
                        {
                            if ( topk_tmp.size() < m_dparam.unit_beam_size || 
                                prob_in[ t*D + it->first ] > topk_min_score )
                            {
                                topk_tmp.emplace_back( it->first, prob_in[ t*D + it->first ] );
                                topk[ it->first ] = std::vector<WordState>();

                                for ( WordState * wstate : wi_trans )
                                {
                                    topk[ it->first ].emplace_back( 
                                                                    it->second,
                                                                    wstate->wwi_score, //先继承词间得分
                                                                    wstate->n_words + 1,  //计数加一
                                                                    wstate->wwi_lm_state, //先继承状态
                                                                    wstate->word_score,
                                                                    wstate->word_lm_state
                                                                );
                                    // 重新计算词间语言模型得分
                                    topk[ it->first ].back().wwi_score += lm_estimate( m_wwi_kenlm_model, 
                                                                            wstate->wwi_lm_state, 
                                                                            m_unit_table->sym( it->first ), 
                                                                            topk[ it->first ].back().wwi_lm_state
                                                                            );
                                }
                                // 更新topk最小概率
                                if ( topk_tmp.size() < m_dparam.unit_beam_size )
                                {
                                    topk_min_score = std::min( topk_min_score, prob_in[ t*D + it->first ] );
                                }
                            }
                        }
                        else 
                        {
                            for ( WordState * wstate : wi_trans )
                            {
                                topk[ it->first ].emplace_back( 
                                                                it->second,
                                                                wstate->wwi_score, //先继承词间得分
                                                                wstate->n_words + 1,  //计数加一
                                                                wstate->wwi_lm_state, //先继承状态
                                                                wstate->word_score,
                                                                wstate->word_lm_state
                                                            );
                                // 重新计算词间语言模型得分
                                topk[ it->first ].back().wwi_score += lm_estimate( m_wwi_kenlm_model, 
                                                                                  wstate->wwi_lm_state, 
                                                                                  m_unit_table->sym( it->first ), 
                                                                                  topk[ it->first ].back().wwi_lm_state
                                                                        );
                            }
                        }
                    }
                    // std::cout << " add root subfixs: " << m_lexicon_trie->getRoot()->children.size() << "\n";
                }

                // c1 = 0;
                // for ( auto it=topk.begin(); it!=topk.end(); it++ )
                // {
                //     c1 += it->second.size();
                // }
                // std::cout << " after add root subfixs: unit size: " << topk.size() << " , state size: " << c1 << "\n";

                /* 第三种情况, 加入blank */
                if ( topk_tmp.size() < m_dparam.unit_beam_size || 
                    prob_in[ t*D + m_dparam.blank_id ] > topk_min_score )
                {
                    topk[ m_dparam.blank_id ] = std::vector<WordState>();
                    topk_tmp.emplace_back( m_dparam.blank_id, prob_in[ t*D + m_dparam.blank_id ] );
                    // std::cout << " add blank: 1\n";
                    if ( topk_tmp.size() < m_dparam.unit_beam_size )
                    {
                        topk_min_score = std::min(topk_min_score, prob_in[ t*D + m_dparam.blank_id ]);
                    }
                }

                /* 第四种情况, 加入和前缀相同, 但不会产生新路径的字符 */
                if ( last_unit_id != -1 && topk.find( last_unit_id ) == topk.end() )
                {
                    if ( topk_tmp.size() < m_dparam.unit_beam_size || 
                         prob_in[ t*D + last_unit_id ] > topk_min_score )
                    {
                        topk[ last_unit_id ] = std::vector<WordState>();
                        topk_tmp.emplace_back( last_unit_id, prob_in[ t*D + last_unit_id ] );
                    }
                    // std::cout << " add same suffix: 1\n";
                }

                // c1 = 0;
                // for ( auto it=topk.begin(); it!=topk.end(); it++ )
                // {
                //     c1 += it->second.size();
                // }
                // std::cout << " after add blank & same subfixs: unit size: " << topk.size() << " , state size: " << c1 << "\n";

                if ( topk_tmp.empty() )
                {
                    std::cerr << "Decode failed! Empty topk?\n";
                    return -1;
                }

                // 对所有候选路径排序
                size_t topk_size = std::min( topk_tmp.size(), m_dparam.unit_beam_size );
                std::partial_sort(
                                    topk_tmp.begin(),
                                    topk_tmp.begin() + topk_size,
                                    topk_tmp.end(),
                                    []( std::pair<int,float> &A, std::pair<int,float> &B )
                                    { return A.second > B.second; }
                                 );

                /*********************************
                 * 开始合并前缀和候选路径
                 *********************************/

                /* 遍历topK个取值 */
                for ( size_t i=0; i<topk_size; i++ )
                {
                    // 得到 unit id 和 log 概率
                    size_t unit_id = topk_tmp[i].first;
                    float unit_prob = topk_tmp[i].second;
                    std::vector<WordState> & suffix_ptrs = topk[unit_id];
                    // float unit_prob = topk_prob[i];
                    if ( unit_prob < 0 || unit_prob > 1 )
                    {
                        std::cerr << "Decode failed! Prefix beam search decoder need inputs after softmax function but got a value:" << unit_prob << "\n";
                        return -1;
                    }

                    // std::cout << "unid id=" << unit_id << " unit=" << m_unit_table->sym(unit_id) << " porb=" << unit_prob << std::endl;
                    
                    unit_prob = ( unit_prob == 0 ? NEG_INF : log( unit_prob ) );

                    if ( unit_id == m_dparam.blank_id )
                    {	
                        /* 如果这是一个blank, 将不会产生新的beam
                         * *b + b > *b, *nb + b > *b
                         */
                        if ( new_beams.find( prev_prefix ) == new_beams.end() )
                        {	/* 如果这个beam还没有被本t的解码得到，加入*/
                            new_beams[prev_prefix] = PrefixBeam();
                            if ( m_unit_kenlm_model != nullptr )
                            {
                                new_beams[prev_prefix].lm_state = prev_beam.lm_state;
                                new_beams[prev_prefix].prob_lm = prev_beam.prob_lm;
                            }
                            new_beams[prev_prefix].word_states = prev_beam.word_states;
                        }
                        /* 更新blank和非blank的概率值 */
                        new_beams[prev_prefix].prob_b = asrdec::log_add(new_beams[prev_prefix].prob_b,
                                                                        prev_beam.prob_b + unit_prob,
                                                                        prev_beam.prob_nb + unit_prob
                                                                    );
                    }
                    else if ( unit_id == last_unit_id )
                    {
                        /* 如果这是一个连续重复的字符,也不会产生新的前缀
                         * *nb + nb > *nb
                         */
                        if ( new_beams.find( prev_prefix ) == new_beams.end() )
                        {	// 如果这个beam不存在，加入一个
                            new_beams[prev_prefix] = PrefixBeam();
                            if ( m_unit_kenlm_model != nullptr )
                            {
                                new_beams[prev_prefix].lm_state = prev_beam.lm_state;
                                new_beams[prev_prefix].prob_lm = prev_beam.prob_lm;
                            }
                            new_beams[prev_prefix].word_states = prev_beam.word_states;
                        }
                        /* */
                        new_beams[prev_prefix].prob_nb = asrdec::log_add(new_beams[prev_prefix].prob_nb,
                                                                         prev_beam.prob_nb + unit_prob );

                        /* 如果这是一个重复但不连续(中间被blank隔开)的字符, 会产生新的前缀 
                         * *b + nb > *nb
                         */
                        // 并且如果后缀存在
                        if ( ! suffix_ptrs.empty() )
                        {
                            std::vector<int> new_prefix(prev_prefix.begin(),prev_prefix.end());
                            // if ( unit_id != m_dparam.silence_id )
                            // {
                            new_prefix.emplace_back( unit_id );
                            // }

                            if ( new_beams.find( new_prefix ) == new_beams.end() )
                            {
                                init_new_prefix_beam( prev_beam, new_beams, new_prefix, pruned_beams,
                                                      prob_in, T, D, t, unit_prob, unit_id);
                                // 更新word状态
                                new_beams[ new_prefix ].word_states = suffix_ptrs;
                            }
                            
                            /* 更新blank和非blank的概率值 */
                            new_beams[new_prefix].prob_nb = asrdec::log_add( new_beams[new_prefix].prob_nb, 
                                                                             prev_beam.prob_b + unit_prob );
                        }
                    }
                    else
                    {
                        /* 如果这是一个不同的字符,无论如何这将产生新的前缀 
                         * *nb + nb > *nb
                        */
                        // 并且如果后缀存在
                        if ( ! suffix_ptrs.empty() )
                        {
                            std::vector<int> new_prefix(prev_prefix.begin(),prev_prefix.end());
                            // if ( unit_id != m_dparam.silence_id )
                            // {
                            new_prefix.emplace_back( unit_id );
                            // }

                            if ( new_beams.find( new_prefix ) == new_beams.end() )
                            {
                                init_new_prefix_beam( prev_beam, new_beams, new_prefix, pruned_beams,
                                                      prob_in, T, D, t, unit_prob, unit_id);
                                /* 更新blank和非blank的概率值 */
                                new_beams[ new_prefix ].word_states = suffix_ptrs;
                            }
                            /* 更新blank和非blank的概率值 */
                            new_beams[new_prefix].prob_nb = asrdec::log_add( new_beams[new_prefix].prob_nb, 
                                                                            prev_beam.prob_b + unit_prob,
                                                                            prev_beam.prob_nb +unit_prob 
                                                                            );
                        }
                    }
                }
            }
        
            /* 计算beam的总分并完成剪枝 
             * 语言模型得分将选取得分最高的那个单词序列的得分作为beam的得分
             * 对单词序列进行剪枝
             */
            for ( auto it = new_beams.begin(); it != new_beams.end(); it++ )
            {
                /* 计算总得分 */ 
                it->second.prob_total = log_add(it->second.prob_b, it->second.prob_nb)/(t+1-skipped_frames);
                /* 取概率最大值作为语言模型得分(并做长度正则化) 
                 * 如果没有解码出来任何单词,公平起见,给它一个<unk>概率
                 */
                if ( m_unit_kenlm_model != nullptr )
                {   
                    /* 对于空前缀用unk概率 */ 
                    if ( it->first.empty() )
                    {
                        it->second.prob_lm = m_unit_unk_lm_score;
                        it->second.prob_total += m_dparam.unit_lm_weight * it->second.prob_lm;
                    }
                    else
                    {
                        it->second.prob_total += m_dparam.unit_lm_weight * (it->second.prob_lm/it->first.size());
                    }
                }
                /* 计算词间语言模型得分
                   查找所有词间跳转路径里面得分最高的那个
                 */
                float best_wwi_lm_score = NEG_INF;
                float best_word_lm_score = NEG_INF;
                for ( WordState & wstate: it->second.word_states )
                {
                    if ( wstate.n_words == 0 && m_wwi_unk_lm_score > best_wwi_lm_score )
                    {
                        best_wwi_lm_score = m_wwi_unk_lm_score;
                    }
                    else if ( wstate.wwi_score/wstate.n_words > best_wwi_lm_score )
                    {
                        best_wwi_lm_score = wstate.wwi_score/wstate.n_words;
                    }

                    if ( wstate.n_words <= 1 ) 
                    {
                        if ( m_word_unk_lm_score > best_word_lm_score )
                        { best_word_lm_score = m_word_unk_lm_score; }
                    }
                    else if ( wstate.word_score/(wstate.n_words-1) > best_word_lm_score )
                    {
                        best_word_lm_score = wstate.word_score/(wstate.n_words-1);
                    }
                    //std::cout << wstate.n_words << " , " << best_word_lm_score << "\n";
                }
                //exit(0);
                //std::cout << "best_wwi_lm_score: " << best_wwi_lm_score << " best_word_lm_score: " << best_word_lm_score << "\n";
                it->second.prob_total += m_dparam.wwi_lm_weight * best_wwi_lm_score;
                it->second.prob_total += m_dparam.word_lm_weight * best_word_lm_score;
            }

            // std::cout << "\n";

            new_beams_sorted.clear();
            new_beams_sorted.assign( new_beams.begin(), new_beams.end() );
            /* 排序和剪枝 */
            std::partial_sort( new_beams_sorted.begin(),
                               new_beams_sorted.begin() + std::min( new_beams_sorted.size(), m_dparam.beam_size  ),
                               new_beams_sorted.end(),
                               [](std::pair<std::vector<int>,PrefixBeam> & A, std::pair<std::vector<int>,PrefixBeam> & B)
                               { return A.second.prob_total > B.second.prob_total; }
                            );
            if ( t != T-1 )
            {
                active_beams.clear();
                pruned_beams.clear();
                for ( size_t i=0; i<new_beams_sorted.size(); i++ )
                {   
                    // std::cout << "Prefix: ";
                    // for ( int & uidx : new_beams_sorted[i].first )
                    // {
                    //     std::cout << m_unit_table->sym( uidx ) << " ";
                    // }
                    // std::cout << " Total: " << new_beams_sorted[i].second.prob_total ;
                    // std::cout << " HW: " << ( new_beams_sorted[i].second.prob_hw + new_beams_sorted[i].second.hw_state.score ) ;
                    // std::cout << std::endl;

                    if ( i < m_dparam.beam_size )
                    {
                        active_beams.push_back( new_beams_sorted[i] );
                    }
                    else if ( m_dparam.callback_pruned_beam ) 
                    {
                        pruned_beams[ new_beams_sorted[i].first ] = new_beams_sorted[i].second;
                    }
                }
            }
        }

        /* 保存最后的结果 */
        std::vector<int> & best_prefix = new_beams_sorted.front().first;
        PrefixBeam & best_beam = new_beams_sorted.front().second;

        // std::cout << "best_prefix size: " << best_prefix.size() << "\n"; 
        // for ( int idx : best_prefix )
        // {
        //     std::cout << idx << " ";
        // }
        // std::cout << "\n";

        result_out.unit_ids.clear();
        result_out.units.clear();
        for ( int idx : best_prefix )
        {   
            if ( idx != m_dparam.silence_id )
            {
                result_out.unit_ids.push_back( idx );
                result_out.units.push_back( m_unit_table->sym( idx ) );
            }
        }
        result_out.am_score = log_add( best_beam.prob_b, best_beam.prob_nb );
        result_out.lm_score = best_beam.prob_lm;

        return 0;
    }

    #ifdef _PYBIND11
    int PrefixBeamSearchDecoder::pydecode_mode(const py::array_t<float> &prob_in, DecodeResult &result_out, int mode)
    {
        py::buffer_info buffer = prob_in.request();
        if (buffer.ndim != 2)
        {
            std::cerr << "prob_in must a 2-d array\n";
            return -1;
        }
        float *data_ptr = (float *) buffer.ptr;
        if ( mode == 0 ){ return decode0(data_ptr, buffer.shape[0], buffer.shape[1], result_out); }
        else if ( mode == 1 ){ return decode1(data_ptr, buffer.shape[0], buffer.shape[1], result_out); }
        else if ( mode == 2 ){ return decode2(data_ptr, buffer.shape[0], buffer.shape[1], result_out); }
        else if ( mode == 3 ){ return decode3(data_ptr, buffer.shape[0], buffer.shape[1], result_out); }
        else if ( mode == 4 ){ return decode4(data_ptr, buffer.shape[0], buffer.shape[1], result_out); }
        else 
        { 
            std::cerr << "Decode failed! Unknown decode mode!\n";
            return -1;
        }
    }
    #endif

} // namespace pbsd
} // namespace asedec 
