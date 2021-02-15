// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_variadic_split.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include "vpu/ngraph/utilities.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset3.hpp"

#include <memory>
#include <numeric>
#include <ngraph/validation_util.hpp>

namespace vpu {

void dynamicToStaticShapeVariadicSplit(std::shared_ptr<ngraph::Node> target) {
    const auto dsr = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(target->input_value(0).get_node_shared_ptr());
    VPU_THROW_UNLESS(dsr, "DynamicToStaticShape transformation for {} of type {} expects {} as input with index {}",
                     target->get_friendly_name(), target->get_type_info(), ngraph::vpu::op::DynamicShapeResolver::type_info, 0);

    const auto axis_node = ngraph::as_type_ptr<ngraph::opset3::Constant>(target->input_value(1).get_node_shared_ptr());
    VPU_THROW_UNLESS(axis_node, "dynamicToStaticShapeVariadic transformation is not applicable for {}, dynamic axis is not supported", target);

    const auto data_rank = target->get_input_partial_shape(0).rank();
    VPU_THROW_UNLESS(data_rank.is_static(), "dynamicToStaticShapeVariadic transformation for {} doesn't support dynamic rank", target);

    int64_t axis = ngraph::normalize_axis(target->description(), axis_node->cast_vector<int64_t>()[0], data_rank);

    const auto split_lengths_node = ngraph::as_type_ptr<ngraph::opset3::Constant>(target->input_value(2).get_node_shared_ptr());
    VPU_THROW_UNLESS(split_lengths_node, "dynamicToStaticShapeVariadic transformation is not applicable for {}, dynamic split_length is not supported", target);
    const auto split_lengths = split_lengths_node->cast_vector<int64_t>();

    for (const auto & i : split_lengths) {
        VPU_THROW_UNLESS(i != -1, "dynamicToStaticShapeVariadic transformation is not applicable for {}, split_length with -1 is not supported", target);
        VPU_THROW_UNLESS(i > 0, "dynamicToStaticShapeVariadic transformation is not applicable for {}, non-positive split_length  is not supported", target);
    }

    const auto data_shape = dsr->input_value(1).get_node_shared_ptr();
    const auto copied = target->clone_with_new_inputs(target->input_values());
    const auto data_rank_value = data_rank.get_length();
    ngraph::OutputVector first_shape_part, second_shape_part;
    if (axis) {
        first_shape_part.push_back(gatherShapeElements(data_shape, 0, axis));
    }
    if (axis + 1 < data_rank_value) {
        second_shape_part.push_back(gatherShapeElements(data_shape, axis + 1, data_rank_value - axis - 1));
    }
    for (size_t i = 0; i < split_lengths.size(); ++i) {
        const auto dim = ngraph::opset3::Constant::create(data_shape->get_element_type(), {1}, {split_lengths[i]});
        auto dsrShapeInput = dim->shared_from_this();

        if (!first_shape_part.empty() || !second_shape_part.empty()) {
            ngraph::OutputVector output_dims{dim};
            output_dims.insert(output_dims.begin(), first_shape_part.begin(), first_shape_part.end());
            output_dims.insert(output_dims.end(), second_shape_part.begin(), second_shape_part.end());
            dsrShapeInput = std::make_shared<ngraph::opset3::Concat>(output_dims, 0);
        }

        const auto outDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied->output(i), dsrShapeInput);
        outDSR->set_friendly_name(target->get_friendly_name() + "." + std::to_string(i));
        target->output(i).replace(outDSR);
    }
}

}  // namespace vpu
