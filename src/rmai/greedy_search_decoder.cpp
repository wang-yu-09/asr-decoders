/*
 *  Copyright (c) Streamax Tech. and its affiliates.
 *  Authors:
 *         (1) Wang, Yu  Oct 25, 2023
*/
#include "greedy_search_decoder.hpp"

namespace asrdec{
namespace gsd
{
    int GreedyDecoder::init(asrdec::DecodeParams &dparam)
    {
        // 创建UnitTable
        m_unit_table.reset(new asrdec::UnitTable());
        m_unit_table->build( dparam.unit_file );

        // 将参数备份
        m_dparam = dparam;

        return 0;
    }

    int GreedyDecoder::decode(const float *prob_in, size_t T, size_t D, asrdec::DecodeResult &result_out)
    {
        std::vector<size_t> tmp;
        /* 第一遍过滤重复 */
        size_t max_idx = 0;
        float max_value = 0;
        for ( size_t i=0; i<T; i++ )
        {
            /* 查找本帧的最大值 */
            max_idx = 0;
            max_value = prob_in[i * D];
            for (size_t j=1; j<D; j++)
            {
                if ( prob_in[i * D + j] > max_value )
                {
                    max_idx = j;
                    max_value = prob_in[i * D + j];
                }
            }
            /**/
            if ( tmp.empty() || max_idx !=tmp.back() )
            {
                tmp.push_back( max_idx );
            }
        }
        /*第二遍去掉blank*/
        result_out.clear();
        for (size_t & idx : tmp)
        {
            if ( idx != m_dparam.blank_id )
            {
                result_out.unit_ids.push_back( idx );
                result_out.units.push_back( m_unit_table->sym( idx ) );
                //std::cout << m_unit_table->sym( idx ).size() << "\n";
            }
        }

        return 0;
    }

} // namespace gsd

} // namespace asedec