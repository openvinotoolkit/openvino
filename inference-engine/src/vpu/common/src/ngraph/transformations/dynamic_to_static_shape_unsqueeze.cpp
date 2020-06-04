// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_unsqueeze.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/validation_util.hpp"

#include "ngraph/opsets/opset3.hpp"
#include <algorithm>
#include <vector>
#include <memory>

namespace vpu {

void dynamicToStaticShapeUnsqueeze(std::shared_ptr<ngraph::Node> target) {
    const auto dsr = target->input_value(0).get_node_shared_ptr();
    VPU_THROW_UNLESS(std::dynamic_pointer_cast<ngraph::vpu::op::DynamicShapeResolver>(dsr),
        "DynamicToStaticShape transformation for {} of type {} expects {} as input with index {}",
        target->get_friendly_name(), target->get_type_info(), ngraph::vpu::op::DynamicShapeResolver::type_info, 0);

    const auto axes = std::dynamic_pointer_cast<ngraph::opset3::Constant>(target->input_value(1).get_node_shared_ptr());
    VPU_THROW_UNLESS(axes, "DynamicToStaticShape transformation for {} of type {} expects {} as input with index {}",
        target->get_friendly_name(), target->get_type_info(), ngraph::op::Constant::type_info, 1);

    const auto unsqueeze = std::dynamic_pointer_cast<ngraph::opset3::Unsqueeze>(target);
    const auto copied = unsqueeze->clone_with_new_inputs(target->input_values());
    const auto shape = dsr->input(1).get_source_output();

    const auto input_rank = unsqueeze->get_input_partial_shape(0).rank();
    VPU_THROW_UNLESS(input_rank.is_static(), "DynamicToStaticShape transformation for {} expects static input rank, but it is not", target);

    const auto original_axes = axes->cast_vector<int64_t>();

    auto axes_value = ngraph::normalize_axes(
            unsqueeze->description(), original_axes, input_rank + original_axes.size());
    std::sort(axes_value.begin(), axes_value.end());

    const auto rank_value = input_rank.get_length();

    ngraph::OutputVector new_shape_dims;
    if (rank_value) {
        const auto split_axis = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::i64, ngraph::Shape{}, std::vector<int64_t>{0});
        const auto split = std::make_shared<ngraph::opset3::Split>(shape, split_axis, rank_value);
        new_shape_dims = split->outputs();
    }
    // for scalar case -- there is no need to split shape as it is empty

    for (const auto & i : axes_value) {
        const auto new_dim = std::make_shared<ngraph::opset3::Constant>(
                shape.get_element_type(), ngraph::Shape{1}, std::vector<int64_t>{1});
        new_shape_dims.insert(new_shape_dims.begin() + i, new_dim);
    }
    const auto unsqueeze_output_shape = std::make_shared<ngraph::opset3::Concat>(new_shape_dims, 0);
    auto outDsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied, unsqueeze_output_shape);
    outDsr->set_friendly_name(target->get_friendly_name());
    ngraph::replace_node(std::move(target), outDsr);
}

}  // namespace vpu
