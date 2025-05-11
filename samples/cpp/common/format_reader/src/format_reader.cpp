// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>

// clang-format off
#include "bmp.h"
#include "npy.h"
#include "MnistUbyte.h"
#include "yuv_nv12.h"
#include "opencv_wrapper.h"

#include "format_reader.h"
// clang-format on

using namespace FormatReader;

std::vector<Registry::CreatorFunction> Registry::_data;

Register<MnistUbyte> MnistUbyte::reg;
Register<YUV_NV12> YUV_NV12::reg;
Register<NumpyArray> NumpyArray::reg;
#ifdef USE_OPENCV
Register<OCVReader> OCVReader::reg;
#else
Register<BitMap> BitMap::reg;
#endif

Reader* Registry::CreateReader(const char* filename) {
    for (auto maker : _data) {
        Reader* ol = maker(filename);
        if (ol != nullptr && ol->size() != 0)
            return ol;
        if (ol != nullptr)
            delete ol;
    }
    return nullptr;
}

void Registry::RegisterReader(CreatorFunction f) {
    _data.push_back(f);
}

Reader* FormatReader::CreateFormatReader(const char* filename) {
    return Registry::CreateReader(filename);
}
