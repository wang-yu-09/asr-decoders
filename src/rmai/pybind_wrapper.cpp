/*
 *  Python Interface using pybind11
 *  Copyright (c) Streamax Tech. and its affiliates.
 *  Authors:
 *         (1) Wang, Yu  Oct 25, 2023
 */
#ifdef _PYBIND11
/* Python Interface 
 * 这个python接口是基于pybind的，如果你想使用这个python接口
 * 1. 安装pybind: pip install pybind11
 * 2. 编译: 参考我给的编译指令
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "decoder_utils.hpp"
#include "decoder_arks.hpp"
#include "greedy_search_decoder.hpp"
#include "prefix_beam_search_decoder.hpp"
//#include "prefix_beam_search_lexicon_decoder.hpp"
#include "lexicon_transducer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rmai_decoder, m)
{
    py::class_<asrdec::UnitTable>(m, "UnitTable")
        .def(py::init())
        .def("build", &asrdec::UnitTable::build, "build unit table", py::arg("dparam"), py::arg("unk_symbol")="<unk>" )
        .def("idx", &asrdec::UnitTable::idx, "Get index by symbol", py::arg("sym"))
        .def("sym", &asrdec::UnitTable::sym, "Get symbol by index", py::arg("idx"))
        .def("size", &asrdec::UnitTable::size);

    py::class_<asrdec::DecodeParams>(m, "DecodeParams")
        .def(py::init())
        .def_readwrite("lexiconp_file", &asrdec::DecodeParams::lexiconp_file)
        .def_readwrite("lexicon_file", &asrdec::DecodeParams::lexicon_file)
        .def_readwrite("unit_file", &asrdec::DecodeParams::unit_file)
        .def_readwrite("unk_symbol", &asrdec::DecodeParams::unk_symbol)
        .def_readwrite("unit_kenlm_file", &asrdec::DecodeParams::unit_kenlm_file)
        .def_readwrite("word_kenlm_file", &asrdec::DecodeParams::word_kenlm_file)
        .def_readwrite("wwi_kenlm_file", &asrdec::DecodeParams::wwi_kenlm_file)
        .def_readwrite("unit_beam_size", &asrdec::DecodeParams::unit_beam_size)
        .def_readwrite("word_beam_size", &asrdec::DecodeParams::word_beam_size)
        .def_readwrite("beam_size", &asrdec::DecodeParams::beam_size)
        .def_readwrite("blank_id", &asrdec::DecodeParams::blank_id)
        .def_readwrite("silence_id", &asrdec::DecodeParams::silence_id)
        .def_readwrite("unit_lm_weight", &asrdec::DecodeParams::unit_lm_weight)
        .def_readwrite("wwi_lm_weight", &asrdec::DecodeParams::wwi_lm_weight)
        .def_readwrite("word_lm_weight", &asrdec::DecodeParams::word_lm_weight)
        .def_readwrite("callback_pruned_beam", &asrdec::DecodeParams::callback_pruned_beam)
        .def_readwrite("hotword_file", &asrdec::DecodeParams::hotword_file)
        .def_readwrite("hotword_extra_lexicon_file", &asrdec::DecodeParams::hotword_extra_lexicon_file)
        .def_readwrite("max_silence_frames", &asrdec::DecodeParams::max_silence_frames)
        .def_readwrite("truncate_silence", &asrdec::DecodeParams::truncate_silence);

    py::class_<asrdec::DecodeResult>(m, "DecodeResult")
        .def(py::init())
        .def_readonly("unit_ids", &asrdec::DecodeResult::unit_ids)
        .def_readonly("units", &asrdec::DecodeResult::units)
        .def_readonly("words", &asrdec::DecodeResult::words)
        .def_readonly("am_score", &asrdec::DecodeResult::am_score)
        .def_readonly("lm_score", &asrdec::DecodeResult::lm_score);

    py::class_<asrdec::DecodeResultNbest>(m, "DecodeResultNbest")
        .def(py::init())
        .def_readonly("unit_ids", &asrdec::DecodeResultNbest::unit_ids)
        .def_readonly("units", &asrdec::DecodeResultNbest::units);

    py::class_<asrdec::gsd::GreedyDecoder>(m, "GreedyDecoder")
        .def(py::init<>())
        .def("init", &asrdec::gsd::GreedyDecoder::init, "Init decoder", \
              py::arg("dparam"))
        .def("decode", &asrdec::gsd::GreedyDecoder::pydecode, "Decode with greedy search algorithm", \
              py::arg("prob_in"), py::arg("result_out"));

    py::class_<asrdec::pbsd::PrefixBeamSearchDecoder>(m, "PrefixBeamSearchDecoder")
        .def(py::init<>())
        .def("init", &asrdec::pbsd::PrefixBeamSearchDecoder::init, "Init decoder", \
              py::arg("dparam"))
        // .def("decode", &asrdec::pbsd::PrefixBeamSearchDecoder::pydecode, "Decode with prefix beamsearch algorithm", \
        //       py::arg("prob_in"), py::arg("result_out"));
        .def("decode", &asrdec::pbsd::PrefixBeamSearchDecoder::pydecode_mode, "Decode with prefix beamsearch algorithm", \
              py::arg("prob_in"), py::arg("result_out"), py::arg("mode") = 0);

//     py::class_<asrdec::pbsld::PrefixBeamSearchLexiconDecoder>(m, "PrefixBeamSearchLexiconDecoder")
//         .def(py::init<>())
//         .def("init", &asrdec::pbsld::PrefixBeamSearchLexiconDecoder::init, "Init decoder", \
//               py::arg("dparam"))
//         .def("decode", &asrdec::pbsld::PrefixBeamSearchLexiconDecoder::pydecode, "Decode with prefix beamsearch lexicon algorithm", \
//               py::arg("prob_in"), py::arg("result_out"));

    py::class_<asrdec::lt::LexiconTransducer>(m, "LexiconTransducer")
        .def(py::init<>())
        .def("init", &asrdec::lt::LexiconTransducer::init, "Init decoder", \
              py::arg("dparam"))
        .def("decode", &asrdec::lt::LexiconTransducer::decode, "Decode with lexicon", \
              py::arg("units"), py::arg("result_out"));
}
#endif