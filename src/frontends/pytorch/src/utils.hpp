// Copyright (C) 2018-2025 Intel Corporation
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

const std::string pytorch_prefix = "[PyTorch Frontend] ";

const std::string& get_pytorch_prefix();

/// \brief Macro to check whether a boolean condition holds.
/// \param COND Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ov::frontend::OpConversionFailure if `cond` is false.
#ifndef PYTORCH_OP_CONVERSION_CHECK
#    define PYTORCH_OP_CONVERSION_CHECK(COND, ...) \
        OPENVINO_ASSERT_HELPER(::ov::frontend::OpConversionFailure, "", (COND), get_pytorch_prefix(), __VA_ARGS__)
#endif

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs, bool allow_complex = false);

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

std::shared_ptr<Node> get_node_axes_range(const NodeContext& context, const Output<Node>& x);

Output<Node> normalize_axis(const NodeContext& context, const Output<Node>& axis, const Output<Node>& input_node);

std::shared_ptr<Node> numel(const NodeContext& context,
                            const Output<Node>& x,
                            element::Type output_type = element::i32);

element::Type convert_dtype(int64_t dtype_value);
bool is_complex_dtype(int64_t pt_type);

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
std::shared_ptr<op::util::FrameworkNode> cast_fw_node(std::shared_ptr<Node> node,
                                                      std::initializer_list<std::string> types);

std::shared_ptr<Node> make_list_construct(const ov::OutputVector& inputs);

bool is_none_node(const Output<Node>& node);

// TODO: Eliminate the need of this function by implementing more accurate custom data type handling
Any simplified_type_interpret(Any type);

void add_exception_to_fw_node(std::shared_ptr<Node> node, const std::string& msg);

bool is_python_scalar_input(const NodeContext& context, size_t index);

void align_eltwise_input_types(const NodeContext& context,
                               Output<Node>& lhs,
                               Output<Node>& rhs,
                               const bool& is_lhs_python_scalar = false,
                               const bool& ir_rhs_python_scalar = false);
void align_output_types(const NodeContext& context, OutputVector& outputs);

std::deque<Output<Node>> get_list_as_outputs(const Output<Node>& start, bool unsqueeze_for_concat = false);

void copy_runtime_info_and_name(const std::shared_ptr<Node>& from,
                                ov::NodeVector to,
                                const ov::NodeVector& additional_rt_info_src = {});

Output<Node> try_constfold(const Output<Node>& x);

Output<Node> get_input_with_floating_type(const NodeContext& context, size_t idx);

Output<Node> get_input_as_i32(const NodeContext& context, size_t idx);

Output<Node> get_input_concat_if_list(const NodeContext& context, size_t idx);

std::tuple<Output<Node>, Output<Node>> get_inputs_with_promoted_types(const NodeContext& context,
                                                                      size_t lhs_idx,
                                                                      size_t rhs_idx);

// helper ops
Output<Node> masked_fill(ov::pass::NodeRegistry& rg,
                         const Output<Node>& data,
                         const Output<Node>& mask,
                         const Output<Node>& value);

Output<Node> masked_select(const NodeContext& context, const Output<Node>& data, const Output<Node>& mask);

Output<Node> flatten(ov::pass::NodeRegistry& rg, const Output<Node>& value, size_t axis);

bool index_tensor_on_list(ov::pass::NodeRegistry& rg,
                          const Output<Node>& data,
                          const ov::OutputVector& indices,
                          const ov::Rank& rank,
                          Output<Node>& new_output,
                          bool& use_input_as_output);

Output<Node> get_complex_shape(const NodeContext& context, const Output<Node>& complex_input);

namespace op {
template <OutputVector (*T)(const NodeContext&), size_t idx = 0>
OutputVector inplace_op(const NodeContext& context) {
    auto translation_res = T(context);
    FRONT_END_OP_CONVERSION_CHECK(translation_res.size() == 1,
                                  "inplace_op function must be used on single output translators");
    context.mutate_input(idx, translation_res[0]);
    return translation_res;
}

template <OutputVector (*T)(const NodeContext&), size_t idx>
OutputVector optional_out(const NodeContext& context) {
    auto translation_res = T(context);
    if (!context.input_is_none(idx)) {
        FRONT_END_OP_CONVERSION_CHECK(translation_res.size() == 1,
                                      "inplace_op function must be used on single output translators");
        context.mutate_input(idx, translation_res[0]);
    }
    return translation_res;
}

template <typename T>
OutputVector translate_1to1_match_1_inputs(const NodeContext& context) {
    num_inputs_check(context, 1, context.get_input_size());
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
    num_inputs_check(context, 1, context.get_input_size());
    auto x = get_input_with_floating_type(context, 0);
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
        !rhs_type.is<type::PyNone>()) {
        align_eltwise_input_types(context,
                                  lhs,
                                  rhs,
                                  is_python_scalar_input(context, 0),
                                  is_python_scalar_input(context, 1));
    }
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
    virtual size_t output_list_size() const override {
        FRONT_END_NOT_IMPLEMENTED(output_list_size);
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
    virtual bool is_input_inlined(size_t index) const override {
        FRONT_END_NOT_IMPLEMENTED(is_input_inlined);
    }
    virtual std::shared_ptr<TorchDecoder> get_inlined_input_decoder(size_t index) const override {
        FRONT_END_NOT_IMPLEMENTED(get_inlined_input_decoder);
    }
    virtual ov::Any get_attribute(const std::string& name) const override {
        FRONT_END_NOT_IMPLEMENTED(get_attribute);
    }
    virtual size_t get_named_input(const std::string& name) const override {
        FRONT_END_NOT_IMPLEMENTED(get_named_input);
    }
    virtual std::unordered_map<std::string, ov::Any> get_rt_info() const override {
        FRONT_END_NOT_IMPLEMENTED(get_rt_info);
    }

private:
    const std::string m_schema = "NONE";
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
