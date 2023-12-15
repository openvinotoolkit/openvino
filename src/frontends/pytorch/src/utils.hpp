// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"

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
                                int bias_input_idx,
                                const std::vector<int>& unsqueeze_dims = {});

Output<Node> reshape_channelwise(const NodeContext& context,
                                 const Output<Node>& data,
                                 const Output<Node>& shape_source);

std::tuple<Output<Node>, Output<Node>> get_shape_rank(const NodeContext& context,
                                                      const Output<Node>& x,
                                                      bool as_scalar = false,
                                                      element::Type output_type = element::i32);

Output<Node> reshape_kernel_for_group(const NodeContext& context, const Output<Node>& kernel, int64_t groups);

std::shared_ptr<Node> get_axes_range(const NodeContext& context, int input_id);

Output<Node> normalize_axis(const NodeContext& context, const Output<Node>& axis, const Output<Node>& input_node);

std::shared_ptr<Node> numel(const NodeContext& context, const Output<Node>& x);

element::Type convert_dtype(int64_t dtype_value);

Output<Node> apply_dtype(const NodeContext& context, size_t dtype_port, const Output<Node>& input_tensor);

op::PadType convert_pad(const std::string& pt_pad);

Output<Node> concat_list_construct(const Output<Node>& input);

/// \brief Checks if input represents empty list.
/// \param input Input to check.
/// \return true if input is empty list, false - if input is non-empty or non-list.
bool is_empty_list(const Output<Node>& input);

OutputVector make_framework_node_ignore_bodies(const NodeContext& context, const std::string& exception);
OutputVector make_framework_node(const NodeContext& context, const std::string& exception);

std::shared_ptr<op::util::FrameworkNode> cast_fw_node(std::shared_ptr<Node> node, const std::string& type);

std::shared_ptr<op::util::FrameworkNode> make_list_construct(const ov::OutputVector& inputs);

bool is_none_node(const Output<Node>& node);

// TODO: Eliminate the need of this function by implementing more accurate custom data type handling
Any simplified_type_interpret(Any type);

void add_exception_to_fw_node(std::shared_ptr<Node> node, const std::string& msg);

element::Type infer_types(const Output<Node>& lhs, const Output<Node>& rhs, bool align_scalars);
void align_eltwise_input_types(const NodeContext& context,
                               Output<Node>& lhs,
                               Output<Node>& rhs,
                               bool align_scalars = false);
void align_output_types(const NodeContext& context, OutputVector& outputs);

std::deque<Output<Node>> get_list_as_outputs(const Output<Node>& start);

void copy_runtime_info_and_name(const std::shared_ptr<Node>& from,
                                ov::NodeVector to,
                                const ov::NodeVector& additional_rt_info_src = {});

namespace op {
template <OutputVector (*T)(const NodeContext&), size_t idx = 0>
OutputVector inplace_op(const NodeContext& context) {
    auto translation_res = T(context);
    FRONT_END_OP_CONVERSION_CHECK(translation_res.size() == 1,
                                  "inplace_op function must be used on single output translators");
    context.mutate_input(idx, translation_res[0]);
    return translation_res;
}

template <typename T>
OutputVector translate_1to1_match_1_inputs(const NodeContext& context) {
    FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(0), "Input should not be None.");
    auto res = context.mark_node(std::make_shared<T>(context.get_input(0)));
    auto out_type = context.get_output_type(0);
    if (out_type.is<element::Type>()) {
        auto dtype = out_type.as<element::Type>();
        if (dtype.is_static() && dtype != res->output(0).get_element_type()) {
            res = context.mark_node(std::make_shared<ov::op::v0::Convert>(res, dtype));
        }
    }
    return {res};
}

template <typename T>
OutputVector translate_1to1_match_1_inputs_with_fp32_type_alignment(const NodeContext& context) {
    FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(0), "Input should not be None.");
    auto x = context.get_input(0);
    // This const only needed for type alignment
    auto dummy_const = context.mark_node(ov::op::v0::Constant::create(element::f32, Shape({}), {0.5}))->output(0);
    align_eltwise_input_types(context, x, dummy_const);
    return {context.mark_node(std::make_shared<T>(x))};
}

template <typename T>
OutputVector translate_1to1_match_2_inputs(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(0) && !context.input_is_none(1), "Inputs should not be None.");
    return {context.mark_node(std::make_shared<T>(context.get_input(0), context.get_input(1)))};
}

template <typename T>
OutputVector translate_1to1_match_2_inputs_align_types(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(0) && !context.input_is_none(1), "Inputs should not be None.");
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    auto lhs_type = context.get_input_type(0);
    auto rhs_type = context.get_input_type(1);
    // If type is string or None, we shouldn't align
    if (!lhs_type.is<type::Str>() && !rhs_type.is<type::Str>() && !lhs_type.is<type::PyNone>() &&
        !rhs_type.is<type::PyNone>())
        align_eltwise_input_types(context, lhs, rhs, true);
    OutputVector res = {context.mark_node(std::make_shared<T>(lhs, rhs))};
    align_output_types(context, res);
    return res;
}

template <typename T, size_t idx = 0>
OutputVector inplace_translate_1to1_match_2_inputs_align_types(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(0) && !context.input_is_none(1), "Inputs should not be None.");
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    // For inplace op we know direction of type alignment
    if (lhs.get_element_type().is_dynamic() || lhs.get_element_type() != rhs.get_element_type())
        rhs = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(rhs, lhs));
    OutputVector res = {context.mark_node(std::make_shared<T>(lhs, rhs))};
    context.mutate_input(idx, res[0]);
    return res;
}

inline OutputVector return_false_scalar(const NodeContext& context) {
    return {context.mark_node(ov::op::v0::Constant::create(element::boolean, Shape{}, {false}))};
}

inline OutputVector skip_node(const NodeContext& context) {
    return {context.get_input(0)};
}

}  // namespace op

class DummyDecoder : public TorchDecoder {
public:
    virtual Any const_input(size_t index) const override {
        FRONT_END_NOT_IMPLEMENTED(const_input);
    }
    virtual const std::vector<size_t>& inputs() const override {
        FRONT_END_NOT_IMPLEMENTED(inputs);
    }
    virtual const std::string& get_input_debug_name(size_t index) const override {
        FRONT_END_NOT_IMPLEMENTED(get_input_debug_name);
    }
    virtual const std::string& get_input_signature_name(size_t index) const override {
        FRONT_END_NOT_IMPLEMENTED(get_input_signature_name);
    }
    virtual PartialShape get_input_shape(size_t index) const override {
        FRONT_END_NOT_IMPLEMENTED(get_input_shape);
    }
    virtual const std::vector<size_t>& get_input_strides(size_t index) const override {
        FRONT_END_NOT_IMPLEMENTED(get_input_strides);
    }
    virtual Any get_input_type(size_t index) const override {
        FRONT_END_NOT_IMPLEMENTED(get_input_type);
    }
    virtual const std::string& get_output_debug_name(size_t index) const override {
        FRONT_END_NOT_IMPLEMENTED(get_output_debug_name);
    }
    virtual PartialShape get_output_shape(size_t index) const override {
        return PartialShape::dynamic();
    }
    virtual Any get_output_type(size_t index) const override {
        FRONT_END_NOT_IMPLEMENTED(get_output_type);
    }
    virtual bool input_is_none(size_t index) const override {
        FRONT_END_NOT_IMPLEMENTED(input_is_none);
    }
    virtual OutputVector try_decode_get_attr() const override {
        FRONT_END_NOT_IMPLEMENTED(try_decode_get_attr);
    }
    virtual OutputVector as_constant() const override {
        FRONT_END_NOT_IMPLEMENTED(as_constant);
    }
    virtual const std::string& as_string() const override {
        FRONT_END_NOT_IMPLEMENTED(as_string);
    }
    virtual const std::string& get_op_type() const override {
        FRONT_END_NOT_IMPLEMENTED(get_op_type);
    }
    virtual const std::string& get_schema() const override {
        return m_schema;
    }
    virtual size_t num_of_outputs() const override {
        FRONT_END_NOT_IMPLEMENTED(num_of_outputs);
    }
    virtual const std::vector<size_t>& outputs() const override {
        FRONT_END_NOT_IMPLEMENTED(outputs);
    }
    virtual size_t output(size_t index) const override {
        FRONT_END_NOT_IMPLEMENTED(output);
    }
    virtual std::shared_ptr<Node> mark_node(std::shared_ptr<Node> ov_node) const override {
        FRONT_END_NOT_IMPLEMENTED(mark_node);
    }
    virtual size_t get_subgraph_size() const override {
        FRONT_END_NOT_IMPLEMENTED(get_subgraph_size);
    }
    virtual void visit_subgraph(std::function<void(std::shared_ptr<TorchDecoder>)> node_visitor) const override {
        FRONT_END_NOT_IMPLEMENTED(visit_subgraph);
    }
    virtual std::shared_ptr<TorchDecoder> get_subgraph_decoder(size_t index) const override {
        FRONT_END_NOT_IMPLEMENTED(get_subgraph_decoder);
    }
    virtual bool may_produce_alias(size_t in_index, size_t out_index) const override {
        FRONT_END_NOT_IMPLEMENTED(may_produce_alias);
    }
    virtual OutputVector inlined_inputs(size_t start_index) const override {
        FRONT_END_NOT_IMPLEMENTED(inlined_inputs);
    }

private:
    const std::string m_schema = "NONE";
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
