// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_reduce.hpp"
#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"

#include "vpu/ngraph/utilities.hpp"
#include <vpu/utils/error.hpp>
#include "ngraph/graph_util.hpp"

#include "ngraph/opsets/opset3.hpp"
#include <memory>
#include <numeric>
#include <ngraph/validation_util.hpp>

namespace vpu {

void dynamicToStaticShapeReduce(std::shared_ptr<ngraph::Node> target) {
    const auto dsr = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(target->input_value(0).get_node_shared_ptr());
    VPU_THROW_UNLESS(dsr, "DynamicToStaticShape transformation for {} of type {} expects {} as input with index {}",
                     target->get_friendly_name(), target->get_type_info(), ngraph::vpu::op::DynamicShapeResolver::type_info, 0);

    VPU_THROW_UNLESS(std::dynamic_pointer_cast<ngraph::op::util::ArithmeticReductionKeepDims>(target) ||
                     std::dynamic_pointer_cast<ngraph::op::util::LogicalReductionKeepDims>(target),
                     "dynamicToStaticShapeReduce transformation expects arithmetic or logical reduction, but it got {} node of type {}",
                     target->get_friendly_name(), target->get_type_info());


    const auto axes_const_node = ngraph::as_type_ptr<ngraph::opset3::Constant>(target->input_value(1).get_node_shared_ptr());
    VPU_THROW_UNLESS(axes_const_node,
                     "dynamicToStaticShapeReduce transformation for {} of type {} expects {} as input with index {}, but it has {} node of type {} instead",
                     target->get_friendly_name(), target->get_type_info(), ngraph::opset3::Constant::type_info, 1,
                     target->input_value(1).get_node_shared_ptr()->get_friendly_name(), target->input_value(1).get_node_shared_ptr()->get_type_info());

    const auto axes = axes_const_node->cast_vector<int64_t>();

    const auto data_rank = target->get_input_partial_shape(0).rank();
    VPU_THROW_UNLESS(data_rank.is_static(), "dynamicToStaticShapeReduce transformation for {} doesn't support dynamic rank", target);
    const auto data_rank_value = data_rank.get_length();

    bool keep_dims = false;
    if (const auto arithmetic_reduce = std::dynamic_pointer_cast<ngraph::op::util::ArithmeticReductionKeepDims>(target)) {
        keep_dims = arithmetic_reduce->get_keep_dims();
    } else if (const auto logical_reduce = std::dynamic_pointer_cast<ngraph::op::util::LogicalReductionKeepDims>(target)) {
        keep_dims = logical_reduce->get_keep_dims();
    } // assertion earlier excluded other variants

    const auto data_shape = dsr->input_value(1);
    ngraph::Output<ngraph::Node> output_shape;
    if (keep_dims) {
        output_shape = std::make_shared<ngraph::opset3::ScatterElementsUpdate>(
                data_shape,
                ngraph::opset3::Constant::create(data_shape.get_element_type(), {axes.size()}, axes),
                ngraph::opset3::Constant::create(data_shape.get_element_type(), {axes.size()}, std::vector<int64_t>(axes.size(), 1)),
                ngraph::opset3::Constant::create(ngraph::element::i64, {1}, {0}));
    } else {
        std::vector<int64_t> range(data_rank_value);
        std::iota(range.begin(), range.end(), 0);
        std::vector<int64_t> indices;
        std::copy_if(range.cbegin(), range.cend(), std::back_inserter(indices),
                [&axes](int64_t i) { return std::find(axes.cbegin(), axes.cend(), i) == axes.cend(); });

        output_shape = std::make_shared<ngraph::opset3::Gather>(
                data_shape,
                ngraph::opset3::Constant::create(data_shape.get_element_type(), {indices.size()}, indices),
                ngraph::opset3::Constant::create(ngraph::element::i64, {1}, {0}));
    }
    const auto copied = target->clone_with_new_inputs(target->input_values());

    auto outDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied, output_shape);
    outDSR->set_friendly_name(target->get_friendly_name());
    ngraph::replace_node(target, std::move(outDSR));
}
}  // namespace vpu
