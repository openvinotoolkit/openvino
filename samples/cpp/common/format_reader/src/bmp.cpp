// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include <fstream>
#include <iostream>

#include "bmp.h"
// clang-format on

using namespace std;
using namespace FormatReader;

BitMap::BitMap(const string& filename) {
    BmpHeader header;
    BmpInfoHeader infoHeader;

    ifstream input(filename, ios::binary);
    if (!input) {
        return;
    }

    input.read(reinterpret_cast<char*>(&header.type), 2);

    if (header.type != 'M' * 256 + 'B') {
        std::cerr << "[BMP] file is not bmp type\n";
        return;
    }

    input.read(reinterpret_cast<char*>(&header.size), 4);
    input.read(reinterpret_cast<char*>(&header.reserved), 4);
    input.read(reinterpret_cast<char*>(&header.offset), 4);

    input.read(reinterpret_cast<char*>(&infoHeader), sizeof(BmpInfoHeader));

    bool rowsReversed = infoHeader.height < 0;
    _width = infoHeader.width;
    _height = abs(infoHeader.height);
    _shape.push_back(_height);
    _shape.push_back(_width);

    if (infoHeader.bits != 24) {
        cerr << "[BMP] 24bpp only supported. But input has:" << infoHeader.bits << "\n";
        return;
    }

    if (infoHeader.compression != 0) {
        cerr << "[BMP] compression not supported\n";
    }

    int padSize = _width & 3;
    char pad[3];
    size_t size = _width * _height * 3;

    _data.reset(new unsigned char[size], std::default_delete<unsigned char[]>());

    input.seekg(header.offset, ios::beg);

    // reading by rows in invert vertically
    for (uint32_t i = 0; i < _height; i++) {
        uint32_t storeAt = rowsReversed ? i : (uint32_t)_height - 1 - i;
        input.read(reinterpret_cast<char*>(_data.get()) + _width * 3 * storeAt, _width * 3);
        input.read(pad, padSize);
    }
}
