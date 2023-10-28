/*
 *  Copyright (c) Streamax Tech. and its affiliates.
 *  Authors:
 *         (1) Wang, Yu  Oct 25, 2023
*/
#include "lexicon_transducer.hpp"

namespace asrdec{
namespace lt
{
    int LexiconTransducer::init(asrdec::DecodeParams &dparam)
    {   
        /* 构建CTC unit id表 */ 
        m_unit_table = std::make_shared<asrdec::UnitTable>();
        m_unit_table->build( dparam.unit_file );

        /* 构建词典树 */ 
        m_lexicon.reset( new asrdec::lt::LexiconTrie() );
        if ( dparam.lexiconp_file.size() > 0 )
        {
            m_lexicon->build( dparam.lexiconp_file, m_unit_table, dparam.unk_symbol, true );
        }
        else if ( dparam.lexicon_file.size() > 0 )
        {
            m_lexicon->build( dparam.lexicon_file, m_unit_table, dparam.unk_symbol, false );
        }
        else
        {
            std::cerr << "Lexicon Transducer need lexicon or lexiconp file!\n";
            return -1;
        }

        /* 构建语言模型 */
        /* 加载语言模型,本解码器必须使用语言模型,不然无法剪枝词典转换后的结果 */
        if ( dparam.word_kenlm_file.size() == 0 )
        {
            std::cerr << "Lexicon Transducer need language model!\n";
            return -1;
        }
        lm::ngram::Config lm_config;
        m_kenlm_model.reset( new lm::ngram::Model(dparam.word_kenlm_file.c_str(), lm_config ) );

        m_dparam = dparam;
        return 0;
    }

    // int LexiconTransducer::decode(const std::vector<std::string> & units, std::vector<std::string> &result_out)
    float LexiconTransducer::decode(const std::vector<std::string> & units, asrdec::DecodeResult &result_out)
    {
        result_out.clear();
        if ( units.size() == 0 ){ return 0; }

        // std::cout << units.size() << " <- units.size()\n";
        // for ( std::string u: units )
        // {
        //     std::cout << u << "\n";
        // }

        /* 初始化一个令牌 */
        std::vector<PassToken> active_tokens;
        active_tokens.emplace_back( 0, m_lexicon->getRoot() );
        active_tokens.back().lm_state = m_kenlm_model->NullContextState();

        std::vector<PassToken> new_tokens;
        std::vector<PassToken> new_tokens_sorted;

        size_t T = units.size();
        std::vector<PassToken> final_tokens;

        /* 开始主循环 */

        // int i = 0;
        while ( ! active_tokens.empty() )
        {
            // std::cout << "\n>>>>>>>>>>>>>>>>>>>> Loop : " << i << " <<<<<<<<<<<<<<<<<<<<<<<<\n";

            new_tokens.clear();

            /* 第一步: 令牌分裂和前传 */

            for ( PassToken & ptoken : active_tokens )
            {
                size_t unit_id = m_unit_table->idx( units[ptoken.frame_id] );

                if ( ptoken.trie_node->children.find( unit_id ) == \
                     ptoken.trie_node->children.end() )
                {
                    /* 如果这个新unit不在树的子节点中, 我们将抛弃这一条单词匹配记录
                    * 但是当且仅当这是root节点时, 输出一个<unk>, 保持指针在根节点的位置不动
                    * 这是因为, 如果这个字符没有匹配到任何一个单词的头部
                    * 我们将他去匹配一个<unk>, 之后的字符就可以重新去匹配单词
                    * 实际上除了使用这种策略匹配到词典中没有的单词的策略
                    * 我们在每个单词匹配的过程中也会分裂出两条beam，其中一条继续前传，而另一条则是武断地截断它
                    * 输出一个中间单词或者unk (我们在构建词典树的时候已经向没有输出符号的词典树节点中插入了unk符号)
                    */
                    if ( ptoken.trie_node == m_lexicon->getRoot() )
                    {   
                        /* 加入一个token */
                        new_tokens.emplace_back( ptoken.frame_id+1, m_lexicon->getRoot() );
                        PassToken & ntoken = new_tokens.back();
                        /* 更新单词序列 */
                        ntoken.words.assign( ptoken.words.begin(), ptoken.words.end() );
                        ntoken.words.push_back( m_dparam.unk_symbol );
                        /* 更新语言模型得分和状态 */
                        ntoken.score = ptoken.score + asrdec::lm_estimate( m_kenlm_model,
                                                                   ptoken.lm_state, 
                                                                   m_dparam.unk_symbol, 
                                                                   ntoken.lm_state
                                                                );
                    }
                }
                else
                {
                    /* 如果这个节点可以前传, 那我们将直接前传到下一个树节点,
                    * 因为我们在事先构造树时,在所有的无输出符号的节点处都插入了一个unk符号,因此所有节点都是有输出符号的
                    * 将token分裂为两批, 一批直接输出符号,然后回退指针
                    * 另一批不输出符号, 继续向前传播(如果还有子节点的话)
                    */
                    std::shared_ptr<asrdec::lt::TrieNode> new_trie_node = ptoken.trie_node->children[unit_id];

                    /* 第一批节点分裂:
                    * 输出标签,然后回退到根节点
                    */
                    for ( std::string & label : new_trie_node->labels )
                    {
                        /* 加入一个新的单词序列 */
                        new_tokens.emplace_back( ptoken.frame_id+1, m_lexicon->getRoot() );
                        PassToken & ntoken = new_tokens.back();
                        /* 更新单词序列 */
                        ntoken.words.assign( ptoken.words.begin(), ptoken.words.end() );
                        ntoken.words.push_back( label );
                        /* 更新语言模型得分和状态 
                        * 如果这是一个unk符号，并且节点上存在词典概率,使用这个概率
                        */
                        ntoken.score = ptoken.score + asrdec::lm_estimate( m_kenlm_model,
                                                                            ptoken.lm_state, 
                                                                            label,
                                                                            ntoken.lm_state);
                    }
                    /* 第二批节点分裂:
                    * 如果还有子节点,并且这不是最后一帧,则不输出标签,继续前传
                    */
                    if ( ! new_trie_node->children.empty() )
                    {
                        new_tokens.emplace_back( ptoken.frame_id+1, new_trie_node );
                        PassToken & ntoken = new_tokens.back();
                        /* 更新单词序列 */
                        ntoken.words.assign( ptoken.words.begin(), ptoken.words.end() );
                        /* 更新语言模型得分和状态 */
                        ntoken.score = ptoken.score;
                        ntoken.lm_state = ptoken.lm_state;
                    }
                }
            }

            /* 第二步,剪枝和过滤 */
            // std::cout << "Prune Before: \n";
            // for ( PassToken & token : new_tokens )
            // {
            //     for ( std::string & word : token.words )
            //     {
            //         std::cout << word << " ";
            //     }
            //     std::cout << " score: " << token.score << " terminated: " << ( token.trie_node == m_lexicon->getRoot() ) << "\n";
            // }

            active_tokens.clear();
            for ( PassToken & token : new_tokens )
            {
                /* 如果这个令牌已经传递到了序列的终点,探索结束,保存
                 * 但如果这个令牌没有回退到根节点(仍然是处于继续前传的状态),抛弃它
                 */
                if ( token.frame_id == T )
                {
                    if ( token.trie_node == m_lexicon->getRoot() )
                    {
                        final_tokens.push_back( token );
                    }
                }
                else
                {
                    active_tokens.push_back( token );
                }
            }

            /* 排序剪枝 */
            if ( active_tokens.size() > m_dparam.word_beam_size )
            {
                new_tokens_sorted.clear();
                new_tokens_sorted.swap( active_tokens );
                std::partial_sort(
                                   new_tokens_sorted.begin(),
                                   new_tokens_sorted.begin() + m_dparam.word_beam_size,
                                   new_tokens_sorted.end(),
                                    [](PassToken & A, PassToken & B)
                                    {
                                        return A.score > B.score;
                                    }
                            );
                active_tokens.clear();
                active_tokens.assign( new_tokens_sorted.begin(), new_tokens_sorted.begin() + m_dparam.word_beam_size );
                //active_tokens.assign( new_tokens_sorted.begin(), new_tokens_sorted.end() );
            }

            // std::cout << "Prune Later: \n";

            // for ( PassToken & token : active_tokens )
            // {
            //     for ( std::string & word : token.words )
            //     {
            //         std::cout << word << " ";
            //     }
            //      std::cout << " score: " << token.score << "\n";
            // }

            // i++;

        }

        if ( final_tokens.empty() )
        {
            std::cerr << "Lexicon Transducer decode failed!\n";
            return -1;
        }

        // std::cout << "\n >>>>>>>>>>>>>>>> final <<<<<<<<<<<<<<<<<<<<<<<< \n";
        // std::cout << "Final Tokens: " << final_tokens.size() << "\n";

        /* 对最后的单词序列取得最佳解码结果 */
        size_t best_idx = 0;
        for ( size_t i=0; i<final_tokens.size(); i++ )
        {
            // final_tokens[i].score = lm_estimate( final_tokens[i].words );
            // for ( std::string & word : final_tokens[i].words )
            // {
            //     std::cout << word << " ";
            // }
            // std::cout << "  score: " << final_tokens[i].score <<"\n";

            if ( final_tokens[i].score > final_tokens[best_idx].score )
            {
                best_idx = i;
            }
        }

        // std::cout << "here2\n";

        result_out.words.clear();
        result_out.words.assign( final_tokens[best_idx].words.begin(),final_tokens[best_idx].words.end() );

        // for ( std::string & word : result_out )
        // {
        //     std::cout << word << " ";
        // }
        // std::cout << "\n";

        return final_tokens[best_idx].score;
    }

    double LexiconTransducer::lm_estimate(std::vector<std::string> &words)
    {
        lm::ngram::State in_state = m_kenlm_model->NullContextState();
        lm::ngram::State out_state;

        double score = 0.0;
        for (std::string & word : words )
        {
            lm::WordIndex vocab = m_kenlm_model->GetVocabulary().Index( word.c_str() );
            lm::FullScoreReturn ret = m_kenlm_model->FullScore(in_state, vocab, out_state);
            score += ret.prob;
            in_state = out_state;
        }
        return score;
    }

} // namespace lt
} // namespace asedec