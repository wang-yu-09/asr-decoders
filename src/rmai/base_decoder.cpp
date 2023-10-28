/*
 *  Copyright (c) Streamax Tech. and its affiliates.
 *  Authors:
 *         (1) Wang, Yu  Oct 25, 2023
*/
#include "base_decoder.hpp"

namespace asrdec{
namespace bd
{
    /*Python Interface*/
    #ifdef _PYBIND11
    int BaseDecoder::pydecode(const py::array_t<float> &prob_in, DecodeResult &result_out)
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

} // namespace bd

} // namespace asedec