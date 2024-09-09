// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/img.hpp"
#include "openvino/reference/utils/bmp.hpp"

namespace ov {
namespace reference {
namespace img {

std::shared_ptr<images> ParserImages(const char* filename) {
    auto ptr = BitMap::getBMP();
    if (ptr->isSupported(filename))
        return ptr;
    return nullptr;
}

}  // namespace img
}  // namespace reference
}  // namespace ov
