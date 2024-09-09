// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/bmp.hpp"

namespace ov {
namespace reference {
namespace img {

std::shared_ptr<BitMap> BitMap::getBMP() {
    if (bmp_singleton == nullptr) {
        auto tmp = std::shared_ptr<BitMap>(new BitMap());
        bmp_singleton = tmp;
    }
    return bmp_singleton;
}

bool BitMap::isSupported(const string& filename) {
    input.open(filename, ios::binary);
    if (!input) {
        return false;
    }

    input.read(reinterpret_cast<char*>(&header.type), 2);

    if (header.type != 'M' * 256 + 'B') {
        std::cerr << "[BMP] file is not bmp type\n";
        return false;
    }

    input.read(reinterpret_cast<char*>(&header.size), 4);
    input.read(reinterpret_cast<char*>(&header.reserved), 4);
    input.read(reinterpret_cast<char*>(&header.offset), 4);

    input.read(reinterpret_cast<char*>(&infoHeader), sizeof(BmpInfoHeader));

    _width = infoHeader.width;
    _height = abs(infoHeader.height);
    _shape.push_back(_height);
    _shape.push_back(_width);
    _shape.push_back(3);

    if (infoHeader.bits != 24) {
        cerr << "[BMP] 24bpp only supported. But input has:" << infoHeader.bits << "\n";
        return false;
    }

    if (infoHeader.compression != 0) {
        cerr << "[BMP] compression not supported\n";
        return false;
    }
    return true;

    // int padSize = _width & 3;
    // char pad[3];
    // size_t size = _width * _height * 3;

    // _data.reset(new unsigned char[size], std::default_delete<unsigned char[]>());

    // input.seekg(header.offset, ios::beg);

    // // reading by rows in invert vertically
    // for (uint32_t i = 0; i < _height; i++) {
    //     uint32_t storeAt = rowsReversed ? i : (uint32_t)_height - 1 - i;
    //     input.read(reinterpret_cast<char*>(_data.get()) + _width * 3 * storeAt, _width * 3);
    //     input.read(pad, padSize);
    // }
}

int BitMap::getData(Tensor& output) {
    BmpHeader header;
    BmpInfoHeader infoHeader;

    if (!input.is_open() ) {
        return -1;
    }

    bool rowsReversed = false;

    int padSize = _width & 3;
    char pad[3];
    // size_t size = _width * _height * 3;

    output.set_shape(_shape);

    char* output_data = output.data<char>();
    input.seekg(header.offset, ios::beg);

    // reading by rows in invert vertically
    for (uint32_t i = 0; i < _height; i++) {
        uint32_t storeAt = rowsReversed ? i : (uint32_t)_height - 1 - i;
        input.read(output_data + _width * 3 * storeAt, _width * 3);
        input.read(pad, padSize);
    }
    return 0;
}

}  // namespace img
}  // namespace reference
}  // namespace ov
