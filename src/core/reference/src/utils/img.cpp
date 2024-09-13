// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/img.hpp"
#include "openvino/reference/utils/bmp.hpp"
#if WITH_JPEG
#include "openvino/reference/utils/jpeg.hpp"
#endif //WITH_JPEG

namespace ov {
namespace reference {
namespace img {

std::shared_ptr<images> ParserImages(const uint8_t* content, size_t img_length) {
    auto bmp = BitMap::getBmp();
    if (bmp->isSupported(content, img_length))
        return bmp;
#if WITH_JPEG
    auto jpeg = Jpeg::getJpeg();
    if (jpeg->isSupported(content, img_length))
        return jpeg;
#endif //WITH_JPEG
    return nullptr;
}

}  // namespace img
}  // namespace reference
}  // namespace ov
