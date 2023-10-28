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
#include "faster_decoder_wrapper.hpp"

namespace py = pybind11;

PYBIND11_MODULE(eesen_decoder, m)
{
    py::class_<tlg::DecodeParams>(m, "DecodeParams")
        .def(py::init())
        .def_readwrite("fst_in", &tlg::DecodeParams::fst_in)
        .def_readwrite("word_file", &tlg::DecodeParams::word_file)
        .def_readwrite("beam", &tlg::DecodeParams::beam)
        .def_readwrite("max_active", &tlg::DecodeParams::max_active)
        .def_readwrite("min_active", &tlg::DecodeParams::min_active)
        .def_readwrite("lattice_beam", &tlg::DecodeParams::lattice_beam)
        .def_readwrite("prune_interval", &tlg::DecodeParams::prune_interval)
        .def_readwrite("determinize_lattice", &tlg::DecodeParams::determinize_lattice)
        .def_readwrite("beam_delta", &tlg::DecodeParams::beam_delta)
        .def_readwrite("hash_ratio", &tlg::DecodeParams::hash_ratio)
        .def_readwrite("prune_scale", &tlg::DecodeParams::prune_scale)
        .def_readwrite("acoustic_scale", &tlg::DecodeParams::acoustic_scale)
        .def_readwrite("allow_partial", &tlg::DecodeParams::allow_partial);

    py::class_<tlg::LinearDecodeResult>(m, "LinearDecodeResult")
        .def(py::init())
        .def_readonly("words", &tlg::LinearDecodeResult::words)
        .def_readonly("alignments", &tlg::LinearDecodeResult::alignments)
        .def_readonly("am_score", &tlg::LinearDecodeResult::am_score)
        .def_readonly("lm_score", &tlg::LinearDecodeResult::lm_score);

    py::class_<tlg::FasterDecoder>(m, "FasterDecoder")
        .def(py::init<>())
        .def("init", &tlg::FasterDecoder::init, "Init decoder", \
              py::arg("dparam"))
        .def("decode", &tlg::FasterDecoder::pydecode, "Decode with faster decoder algorithm", \
              py::arg("prob_in"), py::arg("result_out"));

}
#endif