// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/decodeimg.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/reference/utils/img.hpp"

namespace ov {
namespace reference {

// Implementation of decodeimg.
int decodeimg(const Tensor& input, Tensor& out) {
    int img_length = input.get_byte_size();
    uint8_t* img_content = static_cast<uint8_t*>(input.data());
    int ret = -1;
    auto reader = ov::reference::img::ParserImages(img_content, img_length);
    if (reader != nullptr) {
        ret = reader->getData(out);
        reader->cleanUp();
        auto shape = out.get_shape();
    }
    return ret;
}

}  // namespace reference
}  // namespace ov
