// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#if WITH_GIF
#include "openvino/reference/utils/gif.hpp"

namespace ov {
namespace reference {
namespace img {

int input_callback(GifFileType* gif_file, GifByteType* buf, int size) {
    GIF* gif = static_cast<GIF*>(gif_file->UserData);
    return gif->readData(buf, size);
}

bool GIF::isSupported(const uint8_t* content, size_t length) {
    _data = content;
    _length = length;
    _offset = 0;
    int error_code = D_GIF_SUCCEEDED;
    _gif_file = DGifOpen(this, &input_callback, &error_code);
    if (error_code != D_GIF_SUCCEEDED || DGifSlurp(_gif_file) != GIF_OK || _gif_file->ImageCount <= 0) {
        return false;
    }

    // const int num_frames = gif_file->ImageCount;
    _width = _gif_file->SavedImages[0].ImageDesc.Width;
    _height = _gif_file->SavedImages[0].ImageDesc.Height;
    _channel = 3;

    _shape.clear();
    _shape.push_back(_height);
    _shape.push_back(_width);
    _shape.push_back(_channel);
    return true;
}

int GIF::getData(Tensor& output) {
    if (_data == nullptr) {
        return -1;
    }

    output.set_shape(_shape);
    uint8_t* dstdata = (uint8_t*)output.data();
    int num_frames = 1;
    for (int k = 0; k < num_frames; k++) {
        uint8_t* this_dst = dstdata + k * _width * _channel * _height;

        SavedImage* this_image = &_gif_file->SavedImages[k];
        GifImageDesc* img_desc = &this_image->ImageDesc;

        size_t imgLeft = img_desc->Left;
        size_t imgTop = img_desc->Top;
        size_t imgRight = img_desc->Left + img_desc->Width;
        size_t imgBottom = img_desc->Top + img_desc->Height;

        if (img_desc->Left != 0 || img_desc->Top != 0 || img_desc->Width != _width || img_desc->Height != _height) {
            if (k == 0) {
                return -2;
            }

            imgLeft = std::max(imgLeft, size_t(0));
            imgTop = std::max(imgTop, size_t(0));
            imgRight = std::min(imgRight, _width);
            imgBottom = std::min(imgBottom, _height);

            uint8_t* last_dst = dstdata + (k - 1) * _width * _channel * _height;
            for (size_t i = 0; i < _height; ++i) {
                uint8_t* p_dst = this_dst + i * _width * _channel;
                uint8_t* l_dst = last_dst + i * _width * _channel;
                for (size_t j = 0; j < _width; ++j) {
                    p_dst[j * _channel + 0] = l_dst[j * _channel + 0];
                    p_dst[j * _channel + 1] = l_dst[j * _channel + 1];
                    p_dst[j * _channel + 2] = l_dst[j * _channel + 2];
                }
            }
        }

        ColorMapObject* color_map =
            this_image->ImageDesc.ColorMap ? this_image->ImageDesc.ColorMap : _gif_file->SColorMap;
        if (color_map == nullptr) {
            return -3;
        }

        for (size_t i = imgTop; i < imgBottom; ++i) {
            uint8_t* p_dst = this_dst + i * _width * _channel;
            for (size_t j = imgLeft; j < imgRight; ++j) {
                GifByteType color_index =
                    this_image->RasterBits[(i - img_desc->Top) * (img_desc->Width) + (j - img_desc->Left)];

                if (color_index >= color_map->ColorCount) {
                    return -4;
                }

                const GifColorType& gif_color = color_map->Colors[color_index];
                p_dst[j * _channel + 0] = gif_color.Red;
                p_dst[j * _channel + 1] = gif_color.Green;
                p_dst[j * _channel + 2] = gif_color.Blue;
            }
        }
    }

    return 0;
}

}  // namespace img
}  // namespace reference
}  // namespace ov

#endif //WITH_GIF