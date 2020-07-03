// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_matmul.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset3.hpp"

#include <memory>
#include <numeric>

namespace vpu {

void get_normalized_shape(ngraph::Output<ngraph::Node>& shape, size_t actual_rank_value, size_t max_rank_value, bool transpose,
                          const ngraph::element::Type& elementType) {
    if (const unsigned rank_diff = max_rank_value - actual_rank_value) {
        ngraph::OutputVector extended_shape_parts =
                {ngraph::opset3::Constant::create(elementType, {rank_diff}, std::vector<int64_t>(rank_diff, 1)), shape};
        shape = std::make_shared<ngraph::opset3::Concat>(extended_shape_parts, 0);
    }
    if (transpose) {
        std::vector<int64_t> indices_value(max_rank_value);
        std::iota(indices_value.begin(), indices_value.end(), 0);
        std::iter_swap(indices_value.rbegin(), indices_value.rbegin() + 1);
        const auto indices = ngraph::opset3::Constant::create(ngraph::element::i64, {indices_value.size()}, indices_value);
        const auto axis = ngraph::opset3::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0});
        shape = std::make_shared<ngraph::opset3::Gather>(shape, indices, axis);
    }
}

void dynamicToStaticShapeMatMul(std::shared_ptr<ngraph::Node> target) {
    const auto matmul = ngraph::as_type_ptr<ngraph::opset3::MatMul>(target);
    VPU_THROW_UNLESS(matmul, "dynamicToStaticShapeMatMul transformation is not applicable for {}, it should be {} instead",
            target, ngraph::opset3::MatMul::type_info);

    auto shapeToConstant = [&target](const ngraph::Output<ngraph::Node>& output,
                                     const ngraph::element::Type& elementType) -> std::shared_ptr<ngraph::opset3::Constant> {
        VPU_THROW_UNLESS(output.get_partial_shape().is_static(),
                         "DynamicToStaticShape transformation for {} of type {} expects static shape on inputs without DSR",
                         target->get_friendly_name(), target->get_type_info());
        return ngraph::opset3::Constant::create(elementType, {output.get_shape().size()}, output.get_shape());
    };

    const auto a_input_DSR = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(target->input_value(0).get_node_shared_ptr());
    const auto b_input_DSR = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(target->input_value(1).get_node_shared_ptr());

    if (a_input_DSR && b_input_DSR) {
        VPU_THROW_UNLESS(a_input_DSR->get_input_element_type(1) == b_input_DSR->get_input_element_type(1),
            "DynamicToStaticShape transformation for {} of type {} expects equal shapes data types, actual {} vs {}",
            matmul->get_friendly_name(), matmul->get_type_info(),
            a_input_DSR->get_input_element_type(1), b_input_DSR->get_input_element_type(1));
    }
    VPU_THROW_UNLESS(a_input_DSR || b_input_DSR, "DynamicToStaticShape transformation for {} of type {} expects at least one DSR as input",
                     target->get_friendly_name(), target->get_type_info());

    const auto shapeElementType = a_input_DSR ? a_input_DSR->get_input_element_type(1) : b_input_DSR->get_input_element_type(1);

    ngraph::Output<ngraph::Node> a_input_shape = a_input_DSR ? a_input_DSR->input_value(1) : shapeToConstant(target->input_value(0), shapeElementType);
    ngraph::Output<ngraph::Node> b_input_shape = b_input_DSR ? b_input_DSR->input_value(1) : shapeToConstant(target->input_value(1), shapeElementType);

    const auto& a_rank = a_input_shape.get_partial_shape();
    const auto& b_rank = b_input_shape.get_partial_shape();
    VPU_THROW_UNLESS(a_rank.is_static() && b_rank.is_static(), "DynamicToStaticShape transformation for {} doesn't support dynamic rank", matmul);
    const auto a_rank_value = a_rank[0].get_length();
    const auto b_rank_value = b_rank[0].get_length();
    const auto max_rank_value = std::max(ngraph::Dimension::value_type(2), std::max(a_rank_value, b_rank_value));

    get_normalized_shape(a_input_shape, a_rank_value, max_rank_value, matmul->get_transpose_a(), shapeElementType);
    get_normalized_shape(b_input_shape, b_rank_value, max_rank_value, matmul->get_transpose_b(), shapeElementType);

    ngraph::OutputVector output_dims;
    if (max_rank_value > 2) {
        // batch broadcasting
        const auto max_shape = std::make_shared<ngraph::opset3::Maximum>(a_input_shape, b_input_shape);
        std::vector<int64_t> indices_value(max_rank_value - 2);
        std::iota(indices_value.begin(), indices_value.end(), 0);
        const auto indices = ngraph::opset3::Constant::create(ngraph::element::i64, {indices_value.size()}, indices_value);
        const auto axis = ngraph::opset3::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0});
        const auto batch_dims = std::make_shared<ngraph::opset3::Gather>(max_shape, indices, axis);
        output_dims.push_back(batch_dims);
    }
    const auto input_channels = std::make_shared<ngraph::opset3::Gather>(
            a_input_shape,
            ngraph::opset3::Constant::create(ngraph::element::i64, {1}, {max_rank_value - 2}),
            ngraph::opset3::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0}));
    const auto output_channels = std::make_shared<ngraph::opset3::Gather>(
            b_input_shape,
            ngraph::opset3::Constant::create(ngraph::element::i64, {1}, {max_rank_value - 1}),
            ngraph::opset3::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0}));
    output_dims.push_back(input_channels);
    output_dims.push_back(output_channels);

    const auto output_shape = std::make_shared<ngraph::opset3::Concat>(output_dims, 0);
    const auto copied = target->clone_with_new_inputs(target->input_values());
    auto outDsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied, output_shape);
    outDsr->set_friendly_name(target->get_friendly_name());
    ngraph::replace_node(target, outDsr);
}

}  // namespace vpu
