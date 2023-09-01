// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ov_models/builders.hpp"

namespace ov {
namespace builder {

std::shared_ptr<Node> makeROIPooling(const Output<Node>& input,
                                     const Output<Node>& coords,
                                     const Shape& output_size,
                                     const float spatial_scale,
                                     const ov::helpers::ROIPoolingTypes& roi_pool_type) {
    switch (roi_pool_type) {
        case ov::helpers::ROIPoolingTypes::ROI_MAX:
            return std::make_shared<ov::opset3::ROIPooling>(input, coords, output_size, spatial_scale, "max");
        case ov::helpers::ROIPoolingTypes::ROI_BILINEAR:
            return std::make_shared<ov::opset3::ROIPooling>(input, coords, output_size, spatial_scale, "bilinear");
        default:
            throw std::runtime_error("Incorrect type of ROIPooling operation");
    }
}

}  // namespace builder
}  // namespace ov
