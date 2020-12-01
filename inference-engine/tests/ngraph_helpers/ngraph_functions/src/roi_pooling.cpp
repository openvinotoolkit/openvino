// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeROIPooling(const Output<Node>& input,
                                     const Output<Node>& coords,
                                     const Shape& output_size,
                                     const float spatial_scale,
                                     const ngraph::helpers::ROIPoolingTypes& roi_pool_type) {
    switch (roi_pool_type) {
        case helpers::ROIPoolingTypes::ROI_MAX:
            return std::make_shared<ngraph::opset3::ROIPooling>(input, coords, output_size, spatial_scale, "max");
        case helpers::ROIPoolingTypes::ROI_BILINEAR:
            return std::make_shared<ngraph::opset3::ROIPooling>(input, coords, output_size, spatial_scale, "bilinear");
        default:
            throw std::runtime_error("Incorrect type of ROIPooling operation");
    }
}

}  // namespace builder
}  // namespace ngraph
