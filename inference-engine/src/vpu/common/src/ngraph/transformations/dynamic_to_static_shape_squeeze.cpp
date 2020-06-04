// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_squeeze.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/validation_util.hpp"

#include "ngraph/opsets/opset3.hpp"
#include <algorithm>
#include <memory>

namespace vpu {

void dynamicToStaticShapeSqueeze(std::shared_ptr<ngraph::Node> target) {
    const auto dsr = target->input_value(0).get_node_shared_ptr();
    VPU_THROW_UNLESS(std::dynamic_pointer_cast<ngraph::vpu::op::DynamicShapeResolver>(dsr),
        "DynamicToStaticShape transformation for {} of type {} expects {} as input with index {}",
        target->get_friendly_name(), target->get_type_info(), ngraph::vpu::op::DynamicShapeResolver::type_info, 0);

    const auto axes = std::dynamic_pointer_cast<ngraph::opset3::Constant>(target->input_value(1).get_node_shared_ptr());
    VPU_THROW_UNLESS(axes, "DynamicToStaticShape transformation for {} of type {} expects {} as input with index {}",
        target->get_friendly_name(), target->get_type_info(), ngraph::op::Constant::type_info, 1);

    const auto squeeze = std::dynamic_pointer_cast<ngraph::opset3::Squeeze>(target);
    const auto copied = squeeze->clone_with_new_inputs(target->input_values());
    const auto shape = dsr->input(1).get_source_output();

    const auto input_rank = squeeze->get_input_partial_shape(0).rank();
    VPU_THROW_UNLESS(input_rank.is_static(),
            "DynamicToStaticShape transformation for {} expects static input rank, but it is not", target);

    const auto original_axes = axes->cast_vector<int64_t>();
    VPU_THROW_UNLESS(!original_axes.empty(),
            "DynamicToStaticShape transformation for {} does not support default squeezing which may result in rank dynamism", target);

    const auto axes_value = ngraph::normalize_axes(
            squeeze->description(), original_axes, input_rank);
    const auto rank_value = input_rank.get_length();

    std::vector<int64_t> indices_vector;
    for (auto i = 0; i < rank_value; ++i) {
        if (std::find(axes_value.begin(), axes_value.end(), i) == axes_value.end())
            indices_vector.push_back(i);
    }
    const auto index = std::make_shared<ngraph::opset3::Constant>(
            ngraph::element::i64, ngraph::Shape{indices_vector.size()}, indices_vector);
    const auto axis = std::make_shared<ngraph::opset3::Constant>(
            ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{0});
    const auto squeeze_output_shape = std::make_shared<ngraph::opset3::Gather>(shape, index, axis);
    auto outDsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied, squeeze_output_shape);
    outDsr->set_friendly_name(target->get_friendly_name());
    ngraph::replace_node(std::move(target), outDsr);
}

}  // namespace vpu
