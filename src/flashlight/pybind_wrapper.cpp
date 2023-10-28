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
#include "lexicon_decoder_wrapper.hpp"

namespace py = pybind11;

PYBIND11_MODULE(flashlight_decoder, m)
{
    py::class_<flashlight::DecodeParams>(m, "DecodeParams")
        .def(py::init())
        .def_readwrite("beam_size", &flashlight::DecodeParams::beamSize)
        .def_readwrite("beam_size_token", &flashlight::DecodeParams::beamSizeToken)
        .def_readwrite("beam_threshold", &flashlight::DecodeParams::beamThreshold)
        .def_readwrite("lm_weight", &flashlight::DecodeParams::lmWeight)
        .def_readwrite("word_score", &flashlight::DecodeParams::wordScore)
        .def_readwrite("unk_score", &flashlight::DecodeParams::unkScore)
        .def_readwrite("sil_score", &flashlight::DecodeParams::silScore)
        .def_readwrite("log_add", &flashlight::DecodeParams::logAdd)
        .def_readwrite("unit_dict", &flashlight::DecodeParams::unit_dict)
        .def_readwrite("lexicon", &flashlight::DecodeParams::lexicon)
        .def_readwrite("blank_id", &flashlight::DecodeParams::blank_id)
        .def_readwrite("silence_id", &flashlight::DecodeParams::silence_id)
        .def_readwrite("nbest", &flashlight::DecodeParams::nbest)
        .def_readwrite("unk", &flashlight::DecodeParams::unk)
        .def_readwrite("kenlm_model", &flashlight::DecodeParams::kenlm_model);

    py::class_<flashlight::NbestDecodeResult>(m, "NbestDecodeResult")
        .def(py::init())
        .def_readonly("size", &flashlight::NbestDecodeResult::size)
        .def_readonly("units", &flashlight::NbestDecodeResult::units)
        .def_readonly("words", &flashlight::NbestDecodeResult::words);

    py::class_<flashlight::LexiconDecoderWrapper>(m, "LexiconDecoder")
        .def(py::init<>())
        .def("init", &flashlight::LexiconDecoderWrapper::init, "Init decoder", \
              py::arg("dparam"))
        .def("decode", &flashlight::LexiconDecoderWrapper::pydecode, "Decode with greedy search algorithm", \
              py::arg("prob_in"), py::arg("result_out"));

}
#endif