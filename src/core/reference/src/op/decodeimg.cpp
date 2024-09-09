// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/decodeimg.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/reference/utils/img.hpp"

namespace ov {
namespace reference {

// Implementation of RandomUniform that uses Philox algorithm as inner random unsigned integer generator.
void decodeimg(const Tensor& input, Tensor& out) {
    const char* filename = static_cast<const char*>(input.data());
    std::cout << "input filename : " << filename << std::endl;
    filename = "/home/sgui/tensorflow-1.15/tensorflow/contrib/ffmpeg/testdata/small_100.bmp";
    auto reader = ov::reference::img::ParserImages(filename);
    if (reader == nullptr) {
        std::cout << "Unsupported file :" << filename << std::endl;
    }
    // reader->getData(out);
    reader->closeFile();
}

}  // namespace reference
}  // namespace ov
