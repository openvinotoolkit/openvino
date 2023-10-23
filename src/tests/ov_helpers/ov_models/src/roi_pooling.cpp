// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roi_pooling.hpp"

#include <memory>
#include <vector>

#include "common_test_utils/test_enums.hpp"
#include "openvino/core/node.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeROIPooling(const ov::Output<ov::Node>& input,
                                         const ov::Output<ov::Node>& coords,
                                         const ov::Shape& output_size,
                                         const float spatial_scale,
                                         const ov::test::utils::ROIPoolingTypes& roi_pool_type) {
    switch (roi_pool_type) {
    case ov::test::utils::ROIPoolingTypes::ROI_MAX:
        return std::make_shared<ov::op::v0::ROIPooling>(input, coords, output_size, spatial_scale, "max");
    case ov::test::utils::ROIPoolingTypes::ROI_BILINEAR:
        return std::make_shared<ov::op::v0::ROIPooling>(input, coords, output_size, spatial_scale, "bilinear");
    default:
        throw std::runtime_error("Incorrect type of ROIPooling operation");
    }
}

}  // namespace builder
}  // namespace ngraph
