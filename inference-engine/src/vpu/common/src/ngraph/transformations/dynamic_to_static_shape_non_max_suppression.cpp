// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_non_max_suppression.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset3.hpp"

#include <memory>

namespace vpu {

void dynamicToStaticNonMaxSuppression(std::shared_ptr<ngraph::Node> target) {
    const auto dsr1 = target->input_value(1).get_node_shared_ptr();
    VPU_THROW_UNLESS(std::dynamic_pointer_cast<ngraph::vpu::op::DynamicShapeResolver>(dsr1),
                     "DynamicToStaticShape transformation for {} of type {} expects {} as input with index {}",
                     target->get_friendly_name(), target->get_type_info(), ngraph::vpu::op::DynamicShapeResolver::type_info, 1);

    const auto scores_shape = dsr1->input(1).get_source_output();

    const auto index_num_classes = ngraph::opset3::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{1});
    const auto axis_num_classes = ngraph::opset3::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0});
    const auto num_classes = std::make_shared<ngraph::opset3::Gather>(scores_shape, index_num_classes, axis_num_classes);

    const auto index_num_boxes = ngraph::opset3::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{2});
    const auto axis_num_boxes = ngraph::opset3::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0});
    const auto num_boxes = std::make_shared<ngraph::opset3::Gather>(scores_shape, index_num_boxes, axis_num_boxes);

    VPU_THROW_UNLESS(target->inputs().size() > 2,  "DynamicToStaticShape transformation for {} expects at least 3 inputs", target);
    // originally 3rd input is a scalar of any integer type, so we cast and unsqueeze it to 1D
    const auto max_output_boxes_per_class = std::make_shared<ngraph::opset3::Convert>(std::make_shared<ngraph::opset3::Unsqueeze>(
            target->input_value(2).get_node_shared_ptr(), ngraph::opset3::Constant::create(ngraph::element::i32, {1}, {0})), scores_shape.get_element_type());

    const auto max_output_boxes_overall = std::make_shared<ngraph::opset3::Multiply>(max_output_boxes_per_class, num_classes);
    const auto num_selected_boxes = std::make_shared<ngraph::opset3::Minimum>(num_boxes, max_output_boxes_overall);

    const auto triplet_const = ngraph::opset3::Constant::create(scores_shape.get_element_type(), {1}, std::vector<int64_t>{3});
    const auto output_shape = std::make_shared<ngraph::opset3::Concat>(ngraph::OutputVector{num_selected_boxes, triplet_const}, 0);

    const auto copied = target->clone_with_new_inputs(target->input_values());
    ngraph::replace_node(target, std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied, output_shape));
}

}  // namespace vpu
