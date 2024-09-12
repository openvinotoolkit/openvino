// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/img.hpp"
#include "openvino/reference/utils/bmp.hpp"

namespace ov {
namespace reference {
namespace img {

std::shared_ptr<images> ParserImages(const char* content, size_t img_length) {
    auto ptr = BitMap::getBMP();
    if (ptr->isSupported(content, img_length))
        return ptr;
    return nullptr;
}

}  // namespace img
}  // namespace reference
}  // namespace ov
