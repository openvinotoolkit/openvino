// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_topk.hpp"
#include "vpu/ngraph/operations/static_shape_topk.hpp"
#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"

#include "vpu/ngraph/utilities.hpp"
#include <vpu/utils/error.hpp>
#include "ngraph/graph_util.hpp"

#include "ngraph/opsets/opset3.hpp"
#include <memory>
#include <numeric>
#include <ngraph/validation_util.hpp>

namespace vpu {

void dynamicToStaticShapeTopK(std::shared_ptr<ngraph::Node> target) {
    const auto dsr = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(target->input_value(0).get_node_shared_ptr());
    VPU_THROW_UNLESS(dsr, "DynamicToStaticShape transformation for {} of type {} expects {} as input with index {}",
                     target->get_friendly_name(), target->get_type_info(),
                     ngraph::vpu::op::DynamicShapeResolver::get_type_info_static(), 0);

    const auto topk = ngraph::as_type_ptr<ngraph::opset3::TopK>(target);

    const auto data_rank = target->get_input_partial_shape(0).rank();
    VPU_THROW_UNLESS(data_rank.is_static(), "dynamicToStaticShapeTopK transformation for {} doesn't support dynamic rank", target);
    int64_t axis = topk->get_axis();

    const auto data_shape = dsr->input_value(1).get_node_shared_ptr();
    const auto data_rank_value = data_rank.get_length();
    ngraph::OutputVector first_shape_part, second_shape_part;
    if (axis) {
        first_shape_part.push_back(gatherShapeElements(data_shape, 0, axis));
    }
    if (axis + 1 < data_rank_value) {
        second_shape_part.push_back(gatherShapeElements(data_shape, axis + 1, data_rank_value - axis - 1));
    }

    auto k_0d = target->get_input_node_shared_ptr(1);
    if (target->get_input_element_type(1) != ngraph::element::i64)
        k_0d = std::make_shared<ngraph::opset3::Convert>(k_0d, ngraph::element::i64);

    const auto k_1d = std::make_shared<ngraph::opset3::Unsqueeze>(k_0d, ngraph::opset3::Constant::create(ngraph::element::i64, {1}, {0}));

    ngraph::Output<ngraph::Node> output_shape;
    if (first_shape_part.empty() && second_shape_part.empty()) {
        output_shape = k_1d;
    } else {
        ngraph::OutputVector output_dims{k_1d};
        output_dims.insert(output_dims.begin(), first_shape_part.begin(), first_shape_part.end());
        output_dims.insert(output_dims.end(), second_shape_part.begin(), second_shape_part.end());
        output_shape = std::make_shared<ngraph::opset3::Concat>(output_dims, 0);
    }

    std::shared_ptr<ngraph::Node> new_topk;

    if (ngraph::is_type<ngraph::opset3::Constant>(target->get_input_node_shared_ptr(1)))
        new_topk = target->clone_with_new_inputs(target->input_values());
    else
        new_topk = std::make_shared<ngraph::vpu::op::StaticShapeTopK>(
                target->input_value(0),
                target->input_value(1),
                topk->get_provided_axis(),
                topk->get_mode(),
                topk->get_sort_type(),
                ngraph::element::i32);

    for (auto &output : target->outputs()) {
        const auto outDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(new_topk->output(output.get_index()), output_shape);
        outDSR->set_friendly_name(topk->get_friendly_name() + "." + std::to_string(output.get_index()));
        output.replace(outDSR);
    }
}
}  // namespace vpu
