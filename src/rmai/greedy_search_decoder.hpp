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
#include "base_decoder.hpp"

namespace asrdec{
namespace gsd
{
    class GreedyDecoder: public asrdec::bd::BaseDecoder
    {
        public:
            GreedyDecoder(){}
            ~GreedyDecoder()
            {
                if ( m_unit_table != nullptr )
                { m_unit_table.reset(nullptr); }
            }

        public:
            int init(asrdec::DecodeParams &dparam);
            int decode(const float *prob_in, size_t T, size_t D, asrdec::DecodeResult &result_out);
        
        private:
            asrdec::DecodeParams m_dparam;
            std::unique_ptr<asrdec::UnitTable> m_unit_table;
    };

} // namespace gsd

} // namespace asedec