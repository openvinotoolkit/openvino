// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief JPEG reader
 * \file jpeg.hpp
 */
#pragma once

#include <gif_lib.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "img.hpp"

using namespace std;

namespace ov {
namespace reference {
namespace img {
/**
 * \class Jpeg
 * \brief Reader for jpeg files
 */

class GIF : public images {
private:
    // GifFileType _gifInfo;
    GifFileType* _gif_file;

public:
    GIF() : images() {
        _gif_file = nullptr;
    };

    virtual ~GIF() {
        int error_code;
        if (_gif_file)
            DGifCloseFile(_gif_file, &error_code);
    }

    bool isSupported(const uint8_t* content, size_t img_length) override;

    void cleanUp() override {
        _data = nullptr;
        _offset = 0;
        _length = 0;
    }

    size_t size() const override {
        return _width * _height * _channel;
    }

    int getData(Tensor& output) override;

    // int readData(void* buf, size_t size);
};

int input_callback(GifFileType* gif_file, GifByteType* buf, int size);
// static std::shared_ptr<GIF> gif_singleton = nullptr;

}  // namespace img
}  // namespace reference
}  // namespace ov
