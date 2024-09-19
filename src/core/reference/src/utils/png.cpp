// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#if WITH_PNG

#include "openvino/reference/utils/png.hpp"

namespace ov {
namespace reference {
namespace img {

void ErrorHandler(png_structp png_ptr, png_const_charp msg) {
    PNG* png = static_cast<PNG*>(png_get_io_ptr(png_ptr));
    png->_has_erro = true;
    longjmp(png_jmpbuf(png_ptr), 1);
}

void WarningHandler(png_structp png_ptr, png_const_charp msg) {
}

void input_callback(png_structp png_ptr, png_bytep buf, png_size_t size) {
    images* img = (images*)(png_get_io_ptr(png_ptr));
    img->readData(buf, size);
}

bool PNG::isSupported(const uint8_t* content, size_t length) {
    _data = content;
    _length = length;
    _offset = 0;
    _has_erro = false;
    _channel = 3;

    _png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, this, ErrorHandler, WarningHandler);
    if (!_png_ptr || setjmp(png_jmpbuf(_png_ptr))) {
        return false;
    }

    _info_ptr = png_create_info_struct(_png_ptr);
    if (!_info_ptr || _has_erro) {
        return false;
    }
    png_set_read_fn(_png_ptr, this, input_callback);
    png_read_info(_png_ptr, _info_ptr);

    int bit_depth = 0;
    int color_type = 0;
    unsigned int width, height;
    png_get_IHDR(_png_ptr, _info_ptr, &width, &height, &bit_depth, &color_type, nullptr, nullptr, nullptr);
    if (_has_erro || width <= 0 || height <= 0) {
        return false;
    }
    _width = width;
    _height = height;
    const bool has_tRNS = (png_get_valid(_png_ptr, _info_ptr, PNG_INFO_tRNS)) != 0;
    const bool has_alpha = (color_type & PNG_COLOR_MASK_ALPHA) != 0;
    if (has_alpha || has_tRNS) { 
        png_set_strip_alpha(_png_ptr);
    }

    if (bit_depth > 8)
        png_set_strip_16(_png_ptr);

    png_set_packing(_png_ptr);
    int num_passes = png_set_interlace_handling(_png_ptr);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(_png_ptr);

    const bool is_gray = !(color_type & PNG_COLOR_MASK_COLOR);
    if (is_gray) {
        if (bit_depth < 8) {
            png_set_expand_gray_1_2_4_to_8(_png_ptr);
        }
    }
    if (is_gray)
        png_set_gray_to_rgb(_png_ptr);  
    
    png_read_update_info(_png_ptr, _info_ptr);

    _shape.clear();
    _shape.push_back(_height);
    _shape.push_back(_width);
    _shape.push_back(_channel);
    return true;
}

int PNG::getData(Tensor& output) {
    if (_data == nullptr) {
        return -1;
    }

    output.set_shape(_shape);
    uint8_t* dstdata = (uint8_t*)output.data();

    if (setjmp(png_jmpbuf(_png_ptr))) {
        return false;
    }
    int row_bytes = _channel * _width;
    int num_passes = png_set_interlace_handling(_png_ptr);
    for (int p = 0; p < num_passes; ++p) {
        png_bytep row = dstdata;
        for (int h = _height; h-- != 0; row += row_bytes) {
            png_read_row(_png_ptr, row, nullptr);
        }
    }
    // Marks iDAT as valid.
    png_set_rows(_png_ptr, _info_ptr, png_get_rows(_png_ptr, _info_ptr));
    png_read_end(_png_ptr, _info_ptr);
    return 0;
}

}  // namespace img
}  // namespace reference
}  // namespace ov

#endif //WITH_PNG
