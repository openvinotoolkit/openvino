// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/bmp.hpp"

namespace ov {
namespace reference {
namespace img {

std::shared_ptr<BitMap> BitMap::getBmp() {
    if (bmp_singleton == nullptr) {
        auto tmp = std::shared_ptr<BitMap>(new BitMap());
        bmp_singleton = tmp;
    }
    return bmp_singleton;
}

bool BitMap::isSupported(const uint8_t* content, size_t length) {
    if (content==nullptr || length < 24) {
        return false;
    }
    _data = content;
    _length = length;
    _offset = 0;
    memcpy(reinterpret_cast<char*>(&header.type), _data + _offset, 2);
    _offset += 2;

    if (header.type != 'M' * 256 + 'B') {
        // std::cerr << "[BMP] file is not bmp type\n";
        return false;
    }

    memcpy(reinterpret_cast<char*>(&header.size), _data + _offset, 4);
    _offset += 4;
    memcpy(reinterpret_cast<char*>(&header.reserved), _data + _offset, 4);
    _offset += 4;
    memcpy(reinterpret_cast<char*>(&header.offset), _data + _offset, 4);
    _offset += 4;

    memcpy(reinterpret_cast<char*>(&infoHeader), _data + _offset, sizeof(BmpInfoHeader));
    _offset += sizeof(BmpInfoHeader);

    _width = infoHeader.width;
    _height = abs(infoHeader.height);
    _shape.clear();
    _shape.push_back(_height);
    _shape.push_back(_width);
    _shape.push_back(_channel);

    if (infoHeader.bits != 24) {
        // cerr << "[BMP] 24bpp only supported. But input has:" << infoHeader.bits << "\n";
        return false;
    }

    if (infoHeader.compression != 0) {
        // cerr << "[BMP] compression not supported\n";
        return false;
    }
    return true;
}

int BitMap::getData(Tensor& output) {
    if (_data==nullptr) {
        return -1;
    }

    bool rowsReversed = false;
    int padSize = _width & 3;

    output.set_shape(_shape);

    char* output_data = (char*)output.data();
    _offset = header.offset;

    // reading by rows in invert vertically
    for (uint32_t i = 0; i < _height; i++) {
        uint32_t storeAt = rowsReversed ? i : (uint32_t)_height - 1 - i;
        memcpy(output_data + _width * 3 * storeAt, _data + _offset, _width * 3);
        _offset += _width * 3;
        _offset += padSize;
    }
    return 0;
}

}  // namespace img
}  // namespace reference
}  // namespace ov
