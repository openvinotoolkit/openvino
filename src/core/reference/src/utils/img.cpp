// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/img.hpp"

#include "openvino/reference/utils/bmp.hpp"
#if WITH_JPEG
#    include "openvino/reference/utils/jpeg.hpp"
#endif  // WITH_JPEG
#if WITH_PNG
#    include "openvino/reference/utils/png.hpp"
#endif  // WITH_PNG
#if WITH_GIF
#    include "openvino/reference/utils/gif.hpp"
#endif  // WITH_GIF

namespace ov {
namespace reference {
namespace img {

int images::readData(void* buf, size_t size) {
    if (_data != nullptr) {
        if (size > _length - _offset)
            size = _length - _offset;
        memcpy(buf, _data + _offset, size);
        _offset += size;
        return size;
    }
    return 0;
}

std::shared_ptr<images> ParserImages(const uint8_t* content, size_t img_length) {
#if WITH_JPEG
    auto jpeg = std::shared_ptr<images>(new JPEG());
    if (jpeg->isSupported(content, img_length))
        return jpeg;
#endif  // WITH_JPEG
    auto bmp = std::shared_ptr<images>(new BitMap());
    if (bmp->isSupported(content, img_length))
        return bmp;
#if WITH_PNG
    auto png = std::shared_ptr<images>(new PNG());
    if (png->isSupported(content, img_length))
        return png;
#endif  // WITH_PNG
#if WITH_GIF
    auto gif = std::shared_ptr<images>(new GIF());
    if (gif->isSupported(content, img_length)) {
        return gif;
    }
#endif  // WITH_GIF
        OPENVINO_THROW("ParserImages failed to parser input data to ( BMP",
#if WITH_JPEG
                       ", Jpeg",
#endif //WITH_JPEG
#if WITH_PNG
                       ", PNG",
#endif //WITH_PNG
#if WITH_GIF
                       ", GIF",
#endif //WITH_GIF
                       " )");
    return nullptr;
}

}  // namespace img
}  // namespace reference
}  // namespace ov
