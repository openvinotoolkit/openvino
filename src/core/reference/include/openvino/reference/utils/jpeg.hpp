// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief JPEG reader
 * \file jpeg.hpp
 */
#if WITH_JPEG
#pragma once

#include <memory>
#include <string>
#include "img.hpp"
#include <fstream>
#include <iostream>
#include <jpeglib.h>
#include <setjmp.h>

using namespace std;

namespace ov {
namespace reference {
namespace img {
/**
 * \class Jpeg
 * \brief Reader for jpeg files
 */

class JPEG : public images {
private:
    jpeg_decompress_struct _image_info;
    struct jpeg_error_mgr _jerr;
public:
    // static std::shared_ptr<JPEG> getJPEG();
    JPEG() : images() {};

    virtual ~JPEG() {}

    bool isSupported(const uint8_t* content, size_t img_length) override;

    void cleanUp() override {
        _data=nullptr;
        _offset = 0;
        _length = 0;
        jpeg_destroy_decompress(&_image_info);
    }

    size_t size() const override {
        return _width * _height * _channel;
    }

    int getData(Tensor& output) override;
};
void CatchError(j_common_ptr _image_info);
// static std::shared_ptr<JPEG> jpeg_singleton = nullptr;

}  // namespace img
}  // namespace reference
}  // namespace ov
#endif //WITH_JPEG

