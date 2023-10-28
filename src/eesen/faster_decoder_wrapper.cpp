/*
 *  Copyright (c) Streamax Tech. and its affiliates.
 *  Authors:
 *         (1) Wang, Yu  Oct 25, 2023
*/

#include "faster_decoder_wrapper.hpp"
#include "decodable_matrix.hpp"
#include <iostream>

namespace tlg
{
    int FasterDecoder::init(DecodeParams &dparam)
    {
        // 读取解码图fst
        fst::VectorFst<fst::StdArc> *decode_fst = fst::ReadFstKaldi( dparam.fst_in );
        
        // 读取解码器配置
        eesen::LatticeFasterDecoderConfig dconfig;
        dconfig.beam = dparam.beam;
        dconfig.max_active = dparam.max_active;
        dconfig.min_active = dparam.min_active;
        dconfig.lattice_beam = dparam.lattice_beam;
        dconfig.prune_interval = dparam.prune_interval;
        dconfig.determinize_lattice = dparam.determinize_lattice;
        dconfig.beam_delta = dparam.beam_delta;
        dconfig.hash_ratio = dparam.hash_ratio;
        dconfig.prune_scale = dparam.prune_scale;

        // 初始化解码器实例
        m_decoder_backend.reset(new eesen::LatticeFasterDecoder(*decode_fst, dconfig) );

        // 读取单词表
        m_symbol_table = std::unique_ptr<fst::SymbolTable>( fst::SymbolTable::ReadText(dparam.word_file) );
        if ( m_symbol_table == nullptr )
        {
            std::cerr << "Load word table failed! No such file or file is not readable: " << dparam.word_file << "\n";
            return -1;
        }

        m_dparam = dparam;

        return 0;
    }

    int FasterDecoder::decode(const float *prob_in, size_t T, size_t D, LinearDecodeResult &result_out)
    {
        // 包装概率矩阵
        DecodableMatrixScaled decodable(prob_in, T, D, m_dparam.acoustic_scale);

        // std::cout << "T: " << T << " D: " << D << "\n"; 

        // 解码
        if ( ! m_decoder_backend->Decode( &decodable ) )
        {
            std::cerr << "Decode failed!\n";
            return -1;
        }

        // std::cout << "Decode done!\n";

        // 判断是否有路径到达了图的终止节点
        if ( (! m_decoder_backend->ReachedFinal()) && (! m_dparam.allow_partial) )
        {
            std::cerr << "No any path reached final-state. you can set allow_partial=true to allow partial path.\n";
            return -1;
        }   
        
        // 取出线性结果
        fst::VectorFst<eesen::LatticeArc> decoded;
        m_decoder_backend->GetBestPath( &decoded );

        eesen::LatticeWeight weight;
        std::vector<eesen::int32> alignment;
        std::vector<eesen::int32> words;
        GetLinearSymbolSequence(decoded, &alignment, &words, &weight);

        // 保存结果到输出
        result_out.clear();
        for ( size_t i=0; i<words.size(); i++ )
        {
            std::string s = m_symbol_table->Find( words[i] );
            if ( s == "" ){ s = "<unk>"; }
            result_out.words.push_back( s );
        }
        result_out.am_score = -weight.Value1();
        result_out.lm_score = -weight.Value2();

        return 0;
    }

    /*Python Interface*/
    #ifdef _PYBIND11
    int FasterDecoder::pydecode(const py::array_t<float> &prob_in, LinearDecodeResult &result_out)
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

} // namespace tlg