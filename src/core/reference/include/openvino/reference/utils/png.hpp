// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief JPEG reader
 * \file jpeg.hpp
 */
#pragma once

#include <memory>
#include <string>
#include "img.hpp"
#include <fstream>
#include <iostream>
#include <png.h>

using namespace std;

namespace ov {
namespace reference {
namespace img {
/**
 * \class Jpeg
 * \brief Reader for jpeg files
 */

class PNG : public images {
private:
  png_structp _png_ptr;
  png_infop _info_ptr;

// typedef struct PngInfoHeaderType  {
//   const uint8* data;
//   int data_left;
//   png_structp png_ptr;
//   png_infop info_ptr;
//   png_uint_32 width, height;
//   int num_passes;
//   int color_type;
//   int bit_depth;
//   int channels;
//   bool need_to_synthesize_16;
//   bool error_condition;
//   DecodeContext() : png_ptr(NULL), info_ptr(NULL) {}
// };

public:
    // static std::shared_ptr<PNG> getPNG();
    PNG() : images() {};

    virtual ~PNG() {}

    bool isSupported(const uint8_t* content, size_t img_length) override;

    void cleanUp() override {
        _data=nullptr;
        _offset = 0;
        _length = 0;
          if (_png_ptr) {
    png_destroy_read_struct(&_png_ptr,
                            _info_ptr ? &_info_ptr : nullptr,
                            nullptr);
    _png_ptr = nullptr;
    _info_ptr = nullptr;
  }
    }

    size_t size() const override {
        return _width * _height * _channel;
    }

    int getData(Tensor& output) override;

    bool _has_erro;

};

void input_callback(png_structp png_ptr, png_bytep buf, png_size_t size);

}  // namespace img
}  // namespace reference
}  // namespace ov
