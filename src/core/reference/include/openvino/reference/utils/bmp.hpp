// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief BMP reader
 * \file bmp.hpp
 */
#pragma once

#include <memory>
#include <string>
#include "img.hpp"
#include <fstream>
#include <iostream>
using namespace std;

namespace ov {
namespace reference {
namespace img {
/**
 * \class BitMap
 * \brief Reader for bmp files
 */
class BitMap : public images{
private:
    typedef struct BmpHeaderType {
        unsigned short type = 0u; /* Magic identifier            */
        unsigned int size = 0u;   /* File size in bytes          */
        unsigned int reserved = 0u;
        unsigned int offset = 0u; /* Offset to image data, bytes */
    } BmpHeader;

    typedef struct BmpInfoHeaderType {
        unsigned int size = 0u;               /* Header size in bytes      */
        int width = 0, height = 0;            /* Width and height of image */
        unsigned short planes = 0u;           /* Number of colour planes   */
        unsigned short bits = 0u;             /* Bits per pixel            */
        unsigned int compression = 0u;        /* Compression type          */
        unsigned int imagesize = 0u;          /* Image size in bytes       */
        int xresolution = 0, yresolution = 0; /* Pixels per meter          */
        unsigned int ncolours = 0u;           /* Number of colours         */
        unsigned int importantcolours = 0u;   /* Important colours         */
    } BmpInfoHeader;

    BmpHeader header;
    BmpInfoHeader infoHeader;


public:
    // static std::shared_ptr<BitMap> getBmp();
    BitMap() : images() {_channel=3;};

    virtual ~BitMap() {}
    
    bool isSupported(const uint8_t* content, size_t img_length) override;

    void cleanUp() override {
        _data=nullptr;
        _offset = 0;
        _length = 0;
    }

    size_t size() const override {
        return _width * _height * _channel;
    }

    int getData(Tensor& output) override;
};

// static std::shared_ptr<BitMap> bmp_singleton = nullptr;

}  // namespace img
}  // namespace reference
}  // namespace ov
