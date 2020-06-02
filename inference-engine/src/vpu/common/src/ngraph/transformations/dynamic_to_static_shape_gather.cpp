// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_gather.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset3.hpp"

#include <memory>
#include <numeric>

namespace vpu {

void dynamicToStaticShapeGather(std::shared_ptr<ngraph::Node> target) {
    const auto gather = ngraph::as_type_ptr<ngraph::opset3::Gather>(target);
    VPU_THROW_UNLESS(gather, "dynamicToStaticShapeGather transformation is not applicable for {}, it should be {} instead",
            target, ngraph::opset3::Gather::type_info);

    int64_t axis = gather->get_axis();
    VPU_THROW_UNLESS(axis != std::numeric_limits<int64_t>::max() && axis >= 0,
            "dynamicToStaticShapeGather: Unsupported Gather axis {} for node {}", axis, gather);

    auto shapeToConstant = [&gather](const ngraph::Output<ngraph::Node> & output) -> std::shared_ptr<ngraph::opset3::Constant> {
        VPU_THROW_UNLESS(output.get_partial_shape().is_static(),
                         "DynamicToStaticShape transformation for {} of type {} expects static shape on inputs without DSR",
                         gather->get_friendly_name(), gather->get_type_info());
        return ngraph::opset3::Constant::create(ngraph::element::i64, {output.get_shape().size()}, output.get_shape());
    };

    const auto dataDSR = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(gather->input_value(0).get_node_shared_ptr());
    const auto idxDSR = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(gather->input_value(1).get_node_shared_ptr());

    VPU_THROW_UNLESS(dataDSR || idxDSR, "DynamicToStaticShape transformation for {} of type {} expects at least one DSR as input",
                     gather->get_friendly_name(), gather->get_type_info());

    const auto data_shape = dataDSR ? dataDSR->input_value(1) : shapeToConstant(gather->input_value(0));
    const auto indices_shape = idxDSR ? idxDSR->input_value(1) : shapeToConstant(gather->input_value(1));

    auto copied = target->clone_with_new_inputs(target->input_values());
    copied->set_friendly_name("");


    const auto & data_rank = data_shape.get_partial_shape();
    const auto & indices_rank = indices_shape.get_partial_shape();
    VPU_THROW_UNLESS(data_rank.is_static() && indices_rank.is_static(),
            "DynamicToStaticShape transformation for {} doesn't support dynamic rank", gather);

    const auto data_rank_value = data_rank[0].get_length();
    const auto indices_rank_value = indices_rank[0].get_length();
    ngraph::OutputVector output_dims;
    if (axis) {
        std::vector<int64_t> first_data_shape_part_indices(axis);
        std::iota(first_data_shape_part_indices.begin(), first_data_shape_part_indices.end(), 0);
        const auto first_data_shape_part = std::make_shared<ngraph::opset3::Gather>(
                data_shape,
                ngraph::opset3::Constant::create(ngraph::element::i64, {first_data_shape_part_indices.size()}, first_data_shape_part_indices),
                ngraph::opset3::Constant::create(ngraph::element::i64, {1}, {0}));
        output_dims.push_back(first_data_shape_part);
    }
    if (indices_rank_value)
        output_dims.push_back(indices_shape);
    if (axis + 1 < data_rank_value) {
        std::vector<int64_t> second_data_shape_part_indices(data_rank_value - axis - 1);
        std::iota(second_data_shape_part_indices.begin(), second_data_shape_part_indices.end(), axis + 1);
        const auto second_data_shape_part = std::make_shared<ngraph::opset3::Gather>(
                data_shape,
                ngraph::opset3::Constant::create(ngraph::element::i64, {second_data_shape_part_indices.size()}, second_data_shape_part_indices),
                ngraph::opset3::Constant::create(ngraph::element::i64, {1}, {0}));
        output_dims.push_back(second_data_shape_part);
    }
    const auto output_shape = std::make_shared<ngraph::opset3::Concat>(output_dims, 0);
    auto outDsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied, output_shape);
    outDsr->set_friendly_name(copied->get_friendly_name());
    ngraph::replace_node(target, outDsr);
}

}  // namespace vpu
