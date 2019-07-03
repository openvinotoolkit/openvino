// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <format_reader.h>
#include "bmp.h"
#include "MnistUbyte.h"
#include "opencv_wraper.h"

using namespace FormatReader;

std::vector<Registry::CreatorFunction> Registry::_data;

Register<MnistUbyte> MnistUbyte::reg;
#ifdef USE_OPENCV
Register<OCVReader> OCVReader::reg;
#else
Register<BitMap> BitMap::reg;
#endif

Reader *Registry::CreateReader(const char *filename) {
    for (auto maker : _data) {
        Reader *ol = maker(filename);
        if (ol != nullptr && ol->size() != 0) return ol;
        if (ol != nullptr) ol->Release();
    }
    return nullptr;
}

void Registry::RegisterReader(CreatorFunction f) {
    _data.push_back(f);
}

FORMAT_READER_API(Reader*) CreateFormatReader(const char *filename) {
    return Registry::CreateReader(filename);
}