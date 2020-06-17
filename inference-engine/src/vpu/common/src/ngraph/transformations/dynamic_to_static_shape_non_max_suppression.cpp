// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_non_max_suppression.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset3.hpp"

#include <memory>
#include <vpu/ngraph/operations/static_shape_non_maximum_suppression.hpp>

namespace vpu {

void dynamicToStaticNonMaxSuppression(std::shared_ptr<ngraph::Node> node) {
    auto nms_4 = std::dynamic_pointer_cast<ngraph::op::v4::NonMaxSuppression>(node);
    VPU_THROW_UNLESS(nms_4, "dynamicToStaticNonMaxSuppression transformation for {} of type {} expects {} as node for replacement",
                     node->get_friendly_name(), node->get_type_info(), ngraph::op::v4::NonMaxSuppression::type_info);

    auto staticShapeNMS = std::make_shared<ngraph::vpu::op::StaticShapeNonMaxSuppression>(
            nms_4->input_value(0),
            nms_4->input_value(1),
            nms_4->input_value(2),
            nms_4->input_value(3),
            nms_4->input_value(4),
            nms_4->get_box_encoding(),
            nms_4->get_sort_result_descending(),
            nms_4->get_output_type());

    auto dynamicShapeResolver = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
            staticShapeNMS->output(0), staticShapeNMS->output(1));
    dynamicShapeResolver->set_friendly_name(nms_4->get_friendly_name());

    ngraph::replace_node(std::move(nms_4), std::move(dynamicShapeResolver));
}

}  // namespace vpu
