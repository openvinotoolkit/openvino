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
    std::string roi_pool_method = roi_pool_type == helpers::ROIPoolingTypes::ROI_MAX ? "max" : "bilinear";

    std::shared_ptr<ngraph::Node> roi_pooling;
    roi_pooling = std::make_shared<ngraph::opset3::ROIPooling>(input, coords, output_size, spatial_scale, roi_pool_method);
    return roi_pooling;
}

}  // namespace builder
}  // namespace ngraph
