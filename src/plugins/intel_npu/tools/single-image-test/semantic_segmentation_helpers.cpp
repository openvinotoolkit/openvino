//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "semantic_segmentation_helpers.hpp"
#include "tensor_utils.hpp"

void utils::argMax_channels(const ov::Tensor& tensor, std::vector<uint8_t>& resultArgmax, const ov::Layout& layout) {
    OPENVINO_ASSERT(layout == ov::Layout("NCHW") || layout == ov::Layout("NHWC"),
                    "Unsupported layout: ", layout.to_string());

    const ov::Tensor tensorFP32 = npu::utils::toFP32(tensor);
    const auto dataBuffer = tensorFP32.data<const float>();

    const size_t C = tensorFP32.get_shape()[ov::layout::channels_idx(layout)];
    const size_t H = tensorFP32.get_shape()[ov::layout::height_idx(layout)];
    const size_t W = tensorFP32.get_shape()[ov::layout::width_idx(layout)];

    for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
            float argMax = 0.0f;
            uint8_t clsIdx = std::numeric_limits<uint8_t>::max();
            for (size_t c = 0; c < C; c++) {
                size_t offset;

                if (layout == ov::Layout("NCHW")) {
                    offset = c * H * W + h * W + w;
                } else {
                    offset = h * W * C + w * C + c;
                }

                if (argMax < dataBuffer[offset]) {
                    argMax = dataBuffer[offset];
                    clsIdx = static_cast<uint8_t>(c);
                }
            }
            resultArgmax.push_back(clsIdx);
        }
    }
}
