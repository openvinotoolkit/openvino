// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"

namespace ov {

namespace op {
namespace util {
class FrameworkNode;
}  // namespace util
}  // namespace op

namespace frontend {
namespace pytorch {

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs);

Output<Node> make_optional_bias(const Output<Node>& base_op,
                                const NodeContext& context,
                                size_t bias_input_idx,
                                const std::vector<int>& unsqueeze_dims = {});

Output<ov::Node> reshape_channelwise(const NodeContext& context,
                                     Output<ov::Node> data,
                                     Output<ngraph::Node> shape_source);

std::tuple<Output<Node>, Output<Node>> get_shape_rank(const NodeContext& context,
                                                      const Output<Node>& x,
                                                      bool as_scalar = false,
                                                      element::Type output_type = element::i32);

Output<Node> reshape_kernel_for_group(const NodeContext& context, const Output<Node>& kernel, int64_t groups);

std::shared_ptr<Node> get_axes_range(const NodeContext& context, size_t input_id);

std::shared_ptr<Node> numel(const NodeContext& context, size_t input_id);

element::Type convert_dtype(int64_t dtype_value);
ov::op::PadType convert_pad(const std::string& pt_pad);

std::shared_ptr<Node> concat_list_construct(std::shared_ptr<Node> input);

std::shared_ptr<ov::Model> convert_pytorch_model(std::shared_ptr<TorchDecoder> pytorch_model,
                                                 const TensorMap& external_tensor_map = {});

OutputVector convert_node(NodeContext* context);

std::shared_ptr<ov::op::util::FrameworkNode> cast_fw_node(std::shared_ptr<Node> node, const std::string& type);

// TODO: Elimitate the need of this function by implementing more accurate custom data type handling
Any simplified_type_interpret(Any type);

void align_eltwise_input_types(const NodeContext& context,
                               ov::Output<ov::Node>& lhs,
                               ov::Output<ov::Node>& rhs,
                               bool align_scalars = false);

namespace op {
template <OutputVector (*T)(NodeContext&), size_t idx = 0>
OutputVector inplace_op(NodeContext& context) {
    auto translation_res = T(context);
    FRONT_END_OP_CONVERSION_CHECK(translation_res.size() == 1,
                                  "inplace_op function must be used on single output translators");
    context.mutate_input(idx, translation_res[0]);
    return translation_res;
}

template <typename T>
OutputVector translate_1to1_match_1_inputs(NodeContext& context) {
    auto inputs = context.inputs();
    FRONT_END_OP_CONVERSION_CHECK(inputs.size() >= 1, "Operation has no inputs.");
    for (size_t i = 1; i < inputs.size(); i++) {
        FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(i), "Got more inputs than expected.");
    }
    FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(0), "Input should not be None.");
    return {context.mark_node(std::make_shared<T>(inputs[0]))};
}

template <typename T>
OutputVector translate_1to1_match_2_inputs(NodeContext& context) {
    auto inputs = context.inputs();
    FRONT_END_OP_CONVERSION_CHECK(inputs.size() >= 2, "Operation has less then 2 inputs.");
    for (size_t i = 2; i < inputs.size(); i++) {
        FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(i), "Got more inputs than expected.");
    }
    FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(0) && !context.input_is_none(1), "Inputs should not be None.");
    return {context.mark_node(std::make_shared<T>(inputs[0], inputs[1]))};
}

template <typename T>
OutputVector translate_1to1_match_2_inputs_align_types(NodeContext& context) {
    auto inputs = context.inputs();
    FRONT_END_OP_CONVERSION_CHECK(inputs.size() >= 2, "Operation has less then 2 inputs.");
    for (size_t i = 2; i < inputs.size(); i++) {
        FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(i), "Got more inputs than expected.");
    }
    FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(0) && !context.input_is_none(1), "Inputs should not be None.");
    auto lhs = inputs[0];
    auto rhs = inputs[1];
    align_eltwise_input_types(context, lhs, rhs);
    return {context.mark_node(std::make_shared<T>(lhs, rhs))};
}

inline OutputVector return_false_scalar(NodeContext& context) {
    return {context.mark_node(ov::op::v0::Constant::create(element::boolean, Shape{}, {false}))};
}

inline OutputVector skip_node(NodeContext& context) {
    return {context.get_input(0).get_node_shared_ptr()};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
