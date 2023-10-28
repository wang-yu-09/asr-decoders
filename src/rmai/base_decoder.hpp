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

#ifdef _PYBIND11
/* Python Interface 
 * 这个python接口是基于pybind的，如果你想使用这个python接口
 * 1. 安装pybind: pip install pybind11
 * 2. 编译: 参考我给的编译指令
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
#endif

namespace asrdec{
namespace bd
{
    class BaseDecoder
    {
        public:
            BaseDecoder(){}
            ~BaseDecoder(){}

        public:
            virtual int init(asrdec::DecodeParams &dparam) = 0;
            virtual int decode(const float *prob_in, size_t T, size_t D, asrdec::DecodeResult &result_out) = 0;

            /*Python Interface*/
            #ifdef _PYBIND11
            int pydecode(const py::array_t<float> &prob_in, asrdec::DecodeResult &result_out);
            #endif
    };

} // namespace bd

} // namespace asedec