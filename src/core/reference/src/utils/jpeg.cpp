// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#if WITH_JPEG

#include "openvino/reference/utils/jpeg.hpp"

namespace ov {
namespace reference {
namespace img {

void CatchError(j_common_ptr _image_info) {
    // (*_image_info->err->output_message)(_image_info);
    jmp_buf* jpeg_jmpbuf = reinterpret_cast<jmp_buf*>(_image_info->client_data);
    jpeg_destroy(_image_info);
    longjmp(*jpeg_jmpbuf, 1);
}

bool JPEG::isSupported(const uint8_t* content, size_t length) {
    _data = content;
    _length = length;
    _offset = 0;

    _image_info.err = jpeg_std_error(&_jerr);
    _jerr.error_exit = CatchError;

    jmp_buf jpeg_jmpbuf;
    _image_info.client_data = &jpeg_jmpbuf;
    if (setjmp(jpeg_jmpbuf)) {
        return false;
    }

    jpeg_create_decompress(&_image_info);
    jpeg_mem_src(&_image_info, _data, _length);
    jpeg_read_header(&_image_info, TRUE);

    _channel = std::min(_image_info.num_components, 3);

    switch (_channel) {
    case 1:
        _image_info.out_color_space = JCS_GRAYSCALE;
        break;
    case 3:
        if (_image_info.jpeg_color_space == JCS_CMYK || _image_info.jpeg_color_space == JCS_YCCK) {
            _image_info.out_color_space = JCS_CMYK;
        } else {
            _image_info.out_color_space = JCS_RGB;
        }
        break;
    default:
        return false;
    }
    _image_info.do_fancy_upsampling = true;
    _image_info.scale_num = 1;
    _image_info.scale_denom = 1.0;
    _image_info.dct_method = JDCT_DEFAULT;

    jpeg_calc_output_dimensions(&_image_info);

    _height = static_cast<size_t>(_image_info.output_height);
    _width = static_cast<size_t>(_image_info.output_width);
    _channel = static_cast<size_t>(_image_info.num_components);
    _shape.clear();
    _shape.push_back(_height);
    _shape.push_back(_width);
    _shape.push_back(_channel);
    return true;
}

int JPEG::getData(Tensor& output) {
    if (_data == nullptr) {
        return -1;
    }

    jpeg_start_decompress(&_image_info);
    output.set_shape(_shape);
    uint8_t* dstdata = (uint8_t*)output.data();
    JSAMPLE* output_line = static_cast<JSAMPLE*>(dstdata);
    JSAMPLE* tempdata = nullptr;

    const int stride = _width * _channel * sizeof(JSAMPLE);
    const bool use_cmyk = (_image_info.out_color_space == JCS_CMYK);

    if (use_cmyk) {
        tempdata = new JSAMPLE[_image_info.output_width * 4];
    }

    while (_image_info.output_scanline < _height) {
        int num_lines_read = 0;
        if (use_cmyk) {
            num_lines_read = jpeg_read_scanlines(&_image_info, &tempdata, 1);
            if (num_lines_read > 0) {
                for (size_t i = 0; i < _width; ++i) {
                    int offset = 4 * i;
                    const int c = tempdata[offset + 0];
                    const int m = tempdata[offset + 1];
                    const int y = tempdata[offset + 2];
                    const int k = tempdata[offset + 3];
                    int r, g, b;
                    if (_image_info.saw_Adobe_marker) {
                        r = (k * c) / 255;
                        g = (k * m) / 255;
                        b = (k * y) / 255;
                    } else {
                        r = (255 - k) * (255 - c) / 255;
                        g = (255 - k) * (255 - m) / 255;
                        b = (255 - k) * (255 - y) / 255;
                    }
                    output_line[3 * i + 0] = r;
                    output_line[3 * i + 1] = g;
                    output_line[3 * i + 2] = b;
                }
            }
        } else {
            num_lines_read = jpeg_read_scanlines(&_image_info, &output_line, 1);
        }
        if (num_lines_read == 0) {
            break;
        }
        output_line += stride;
    }
    delete[] tempdata;
    tempdata = nullptr;

    if (_channel == 4) {
        JSAMPLE* scanlineptr = static_cast<JSAMPLE*>(dstdata + static_cast<int64_t>(_height - 1) * stride);
        const JSAMPLE kOpaque = -1;  // All ones appropriate for JSAMPLE.
        const int right_rgb = (_width - 1) * 3;
        const int right_rgba = (_width - 1) * 4;

        for (int y = _height; y-- > 0;) {
            const JSAMPLE* rgb_pixel = scanlineptr + right_rgb;
            JSAMPLE* rgba_pixel = scanlineptr + right_rgba;
            scanlineptr -= stride;
            for (int x = _width; x-- > 0; rgba_pixel -= 4, rgb_pixel -= 3) {
                rgba_pixel[3] = kOpaque;
                rgba_pixel[2] = rgb_pixel[2];
                rgba_pixel[1] = rgb_pixel[1];
                rgba_pixel[0] = rgb_pixel[0];
            }
        }
    }

    jpeg_finish_decompress(&_image_info);
    return 0;
}

}  // namespace img
}  // namespace reference
}  // namespace ov

#endif // WITH_JPEG
