// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "op_table.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/pytorch/decoder.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_promote_types.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/util/log.hpp"
#include "pt_framework_node.hpp"
#include "translate_session.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

using namespace ov::op;

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs) {
    auto num_inputs = context.get_input_size();
    FRONT_END_OP_CONVERSION_CHECK(num_inputs >= min_inputs, "Got less inputs than expected");
    for (auto i = max_inputs; i < num_inputs; i++) {
        FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(i), "Got more inputs than expected.");
    }
}

const std::string& get_pytorch_prefix() {
    return pytorch_prefix;
}

Output<Node> make_optional_bias(const Output<Node>& base_op,
                                const NodeContext& context,
                                int bias_input_idx,
                                const std::vector<int>& unsqueeze_dims) {
    using std::make_shared;

    if (!context.input_is_none(bias_input_idx)) {
        auto bias = context.get_input(bias_input_idx);
        if (!unsqueeze_dims.empty()) {
            auto indices = v0::Constant::create(element::i32, {unsqueeze_dims.size()}, unsqueeze_dims);
            context.mark_node(indices);
            bias = make_shared<v0::Unsqueeze>(bias, indices);
            context.mark_output(bias);
        }
        return make_shared<v1::Add>(context.mark_output(base_op), bias);
    } else {
        return base_op;
    }
}

Output<Node> reshape_channelwise(const NodeContext& context,
                                 const Output<Node>& data,
                                 const Output<Node>& shape_source) {
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(shape_source, element::i32));
    auto input_rank = context.mark_node(std::make_shared<v3::ShapeOf>(input_shape, element::i32));
    auto one_const = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto two_const = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {2}));
    auto tail_shape_rank = context.mark_node(std::make_shared<v1::Subtract>(input_rank, two_const));
    auto tail_shape = context.mark_node(std::make_shared<v3::Broadcast>(one_const, tail_shape_rank));
    auto channels_dim = context.mark_node(std::make_shared<v3::ShapeOf>(data, element::i32));
    auto new_shape =
        context.mark_node(std::make_shared<v0::Concat>(OutputVector{one_const, channels_dim, tail_shape}, 0));

    return context.mark_node(std::make_shared<v1::Reshape>(data, new_shape, false));
}

std::tuple<Output<Node>, Output<Node>> get_shape_rank(const NodeContext& context,
                                                      const Output<Node>& x,
                                                      bool as_scalar,
                                                      element::Type output_type) {
    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(x, output_type));
    Output<Node> rank = context.mark_node(std::make_shared<v3::ShapeOf>(shape, output_type));
    if (as_scalar) {
        auto axis_0 = context.mark_node(v0::Constant::create(output_type, Shape{}, {0}));
        rank = context.mark_node(std::make_shared<v0::Squeeze>(rank, axis_0));
    }
    return std::make_tuple(shape, rank);
}

Output<Node> reshape_kernel_for_group(const NodeContext& context, const Output<Node>& kernel, int64_t groups) {
    using std::make_shared;

    auto axis_0 = v0::Constant::create(element::i32, Shape{}, {0});
    auto groups_const = v0::Constant::create(element::i32, Shape{1}, {groups});
    auto neg_1_const = v0::Constant::create(element::i32, Shape{1}, {-1});

    auto kernel_shape = std::make_shared<v3::ShapeOf>(kernel, element::i32);
    auto c_out_idx = v0::Constant::create(element::i32, Shape{}, {0});
    auto kernel_shape_0 = make_shared<v8::Gather>(kernel_shape, c_out_idx, axis_0);
    auto kernel_shape_0_uns = make_shared<v0::Unsqueeze>(kernel_shape_0, axis_0);
    auto c_out_value = make_shared<v1::Divide>(kernel_shape_0_uns, groups_const);

    auto start = v0::Constant::create(element::i32, Shape{1}, {2});
    auto stop = v0::Constant::create(element::i32, Shape{1}, {std::numeric_limits<int32_t>::max()});
    auto step = v0::Constant::create(element::i32, Shape{1}, {1});
    auto remaining_shape = make_shared<v8::Slice>(kernel_shape, start, stop, step);

    auto new_kernel_shape =
        make_shared<v0::Concat>(OutputVector{groups_const, c_out_value, neg_1_const, remaining_shape}, 0);
    auto res = make_shared<v1::Reshape>(kernel, new_kernel_shape, false);
    context.mark_nodes({axis_0,
                        groups_const,
                        kernel_shape,
                        c_out_idx,
                        kernel_shape_0,
                        kernel_shape_0_uns,
                        c_out_value,
                        start,
                        stop,
                        step,
                        remaining_shape,
                        new_kernel_shape,
                        res});
    return res;
}

std::shared_ptr<Node> get_axes_range(const NodeContext& context, int input_id) {
    auto x = context.get_input(input_id);
    return get_node_axes_range(context, x);
};

std::shared_ptr<Node> get_node_axes_range(const NodeContext& context, const Output<Node>& x) {
    auto start = std::make_shared<v0::Constant>(element::i32, Shape{}, 0);
    auto step = std::make_shared<v0::Constant>(element::i32, Shape{}, 1);
    Output<Node> reduced_rank;
    std::tie(std::ignore, reduced_rank) = get_shape_rank(context, x, true);
    return context.mark_node(std::make_shared<v4::Range>(start, reduced_rank, step, element::i32));
};

Output<Node> normalize_axis(const NodeContext& context, const Output<Node>& axis, const Output<Node>& rank) {
    auto axis_rank = std::make_shared<v1::Add>(axis, rank);
    auto new_axis = std::make_shared<v1::Mod>(axis_rank, rank);

    if (const auto axis_const = ov::util::get_constant_from_source(new_axis)) {
        return context.mark_node(axis_const);
    } else {
        context.mark_nodes({axis_rank, new_axis});
        return new_axis;
    }
}

std::shared_ptr<Node> numel(const NodeContext& context, const Output<Node>& x, element::Type output_type) {
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(x, output_type));
    auto axes = context.mark_node(v0::Constant::create(output_type, Shape({1}), {0}));
    return context.mark_node(std::make_shared<v1::ReduceProd>(input_shape, axes, false));
};

namespace {
const std::unordered_map<int64_t, element::Type> TORCH_TO_OV_TYPE{
    {0, element::u8},
    {1, element::i8},
    {2, element::i16},
    {3, element::i32},
    {4, element::i64},
    {5, element::f16},
    {6, element::f32},
    {7, element::f64},
    {11, element::boolean},
    {12, element::i8},   // quantized i8
    {13, element::u8},   // quantized u8
    {14, element::i32},  // quantized i32
    {15, element::bf16},
};

const std::unordered_map<std::string, PadType> TORCH_AUTO_PAD_TO_OV{{"valid", PadType::VALID},
                                                                    {"same", PadType::SAME_UPPER}};
}  // namespace

element::Type convert_dtype(int64_t pt_type) {
    FRONT_END_OP_CONVERSION_CHECK(TORCH_TO_OV_TYPE.count(pt_type), "Unknown type: ", pt_type);
    return TORCH_TO_OV_TYPE.at(pt_type);
};

Output<Node> apply_dtype(const NodeContext& context, size_t dtype_port, const Output<Node>& input_tensor) {
    if (std::dynamic_pointer_cast<v0::Constant>(
            context.get_input_from_visible_context(dtype_port).get_node_shared_ptr())) {
        auto dtype = convert_dtype(context.const_input<int64_t>(dtype_port));
        return context.mark_node(std::make_shared<v0::Convert>(input_tensor, dtype));
    } else if (const auto& fw_node =
                   cast_fw_node(context.get_input(static_cast<int>(dtype_port)).get_node_shared_ptr(), "prim::dtype")) {
        auto out_tensor = fw_node->input_value(0);
        return context.mark_node(std::make_shared<v1::ConvertLike>(input_tensor, out_tensor));
    } else {
        FRONT_END_OP_CONVERSION_CHECK(false, "Couldn't get dtype input");
    }
    return input_tensor;
};

PadType convert_pad(const std::string& pt_pad) {
    FRONT_END_OP_CONVERSION_CHECK(TORCH_AUTO_PAD_TO_OV.count(pt_pad), "Unknown pad: ", pt_pad);
    return TORCH_AUTO_PAD_TO_OV.at(pt_pad);
};

Output<Node> concat_list_construct(const Output<Node>& input) {
    if (auto list_construct = cast_fw_node(input.get_node_shared_ptr(), "prim::ListConstruct")) {
        auto list_inputs = list_construct->input_values();
        OutputVector node_vector;
        auto zero = v0::Constant::create(element::i32, Shape{}, {0});
        for (size_t i = 0; i < list_inputs.size(); i++) {
            auto node = concat_list_construct(list_inputs[i]);
            auto unsqueezed_node = std::make_shared<v0::Unsqueeze>(node, zero);
            node_vector.push_back(unsqueezed_node);
        }
        return std::make_shared<v0::Concat>(node_vector, 0);
    }
    return input;
}

bool is_empty_list(const Output<Node>& input) {
    if (const auto list_construct = cast_fw_node(input.get_node_shared_ptr(), "prim::ListConstruct")) {
        return list_construct->get_input_size() == 0;
    }
    return false;
}

namespace {
std::shared_ptr<PtFrameworkNode> create_fw_node_with_exception(const NodeContext& context,
                                                               const ov::OutputVector& inputs,
                                                               size_t num_outputs,
                                                               const std::string& exception_message,
                                                               bool skip_subgraphs = false) {
    auto fw_node = std::make_shared<PtFrameworkNode>(context.get_decoder(), inputs, num_outputs, false, skip_subgraphs);
    context.mark_node(fw_node);
    if (!exception_message.empty()) {
        auto attrs = fw_node->get_attrs();
        std::string message = "Exception happened during conversion of operation " + fw_node->get_friendly_name() +
                              " with schema " + context.get_schema() + '\n' + exception_message;
        attrs[PtFrameworkNode::failed_conversion_key] = std::move(message);
        fw_node->set_attrs(attrs);
    }
    return fw_node;
}
}  // namespace

OutputVector make_framework_node_ignore_bodies(const NodeContext& context, const std::string& exception) {
    auto fw_node =
        create_fw_node_with_exception(context, context.inputs(), context.get_output_size() + 1, exception, true);
    return fw_node->outputs();
}

OutputVector make_framework_node(const NodeContext& context, const std::string& exception) {
    auto schema = context.get_schema();
    // TODO: properly process schema to get the actual position of mutable input
    // Hack. Can indicate mutable inputs, but can it be reliable?
    if (schema.find('!') != std::string::npos) {
        // We create additional output for such nodes. It contains new tensor that represents input that was changed.
        auto fw_node =
            create_fw_node_with_exception(context, context.inputs(), context.get_output_size() + 1, exception);
        auto outputs = fw_node->outputs();
        // Usually mutated input index is 0, because it is usually "self" input, so we need to replace this tensor with
        // output we created.
        context.mutate_input(0, outputs.back());
        OPENVINO_DEBUG("Created node with mutated 0 input. Schema:", schema, "\n");
        // For simplification we do not expect such operations to have extra bodies
        FRONT_END_OP_CONVERSION_CHECK(context.get_decoder()->get_subgraph_size() == 0,
                                      "Mutable operation has subgraphs.");
        return outputs;
    }

    // Pay attention to subgraphs that may appear in the node
    std::map<size_t, ParameterVector> inputs_map;
    std::map<size_t, ResultVector> extra_outputs_map;
    std::set<size_t> input_idxs;  // initial inputs
    std::vector<std::shared_ptr<Model>> bodies;
    // We need to remember initial inputs to be able to find extra inputs to body that were created to propagate
    // external context
    size_t num_body_outs = 0;
    auto session = context.get_session();
    for (size_t i = 0; i < context.get_decoder()->get_subgraph_size(); ++i) {
        auto subgraph_decoder = context.get_decoder()->get_subgraph_decoder(i);
        auto inputs = subgraph_decoder->inputs();
        input_idxs.insert(inputs.begin(), inputs.end());
        auto body = context.convert_subgraph(i);
        bodies.push_back(body);
        for (const auto& param : body->get_parameters()) {
            auto input_idx = session->decode_tensor_name(param->output(0));
            inputs_map[input_idx].push_back(param);
        }
        const auto& body_outputs = subgraph_decoder->outputs();
        if (i == 0) {
            num_body_outs = body_outputs.size();
        } else {
            FRONT_END_OP_CONVERSION_CHECK(
                num_body_outs == body_outputs.size(),
                "Number of outputs of this body is different from number of outputs of first body");
        }
        // Some bodies may have mutated inputs which we need to propagate to external context
        auto body_results = body->get_results();
        for (size_t i = num_body_outs; i < body_results.size(); i++) {
            auto out_idx = session->decode_tensor_name(body_results[i]->input(0).get_source_output());
            FRONT_END_OP_CONVERSION_CHECK(extra_outputs_map.count(out_idx) == 0,
                                          "More then one body output with same tensor name.");
            extra_outputs_map[out_idx].push_back(body_results[i]);
        }
    }
    // Number of body outputs can be higher then number of pt node outputs, e.g. in case of loop first body output is
    // condition, we have to skip such outputs.
    auto num_skip_body_outputs =
        num_body_outs > context.get_output_size() ? num_body_outs - context.get_output_size() : 0;

    // We need to reduce number of outputs, because some outputs are outputs from body
    auto fw_node = create_fw_node_with_exception(context,
                                                 context.inputs(),
                                                 context.get_output_size() - num_body_outs + num_skip_body_outputs,
                                                 exception);
    fw_node->set_friendly_name(context.get_op_type());
    for (size_t i = 0; i < bodies.size(); ++i) {
        fw_node->set_function(static_cast<int>(i), bodies[i]);
    }

    // Connect inputs with external context
    for (const auto& input : inputs_map) {
        if (!input_idxs.count(input.first)) {
            auto external_output = context.get_tensor_from_model_or_create_input(input.first);
            fw_node->set_invariant_inputs(external_output, input.second);
        } else {
            auto external_output = context.get_tensor_from_model(input.first);
            if (external_output.get_node()) {
                fw_node->set_invariant_inputs(external_output, input.second);
            }
        }
    }
    OutputVector res(context.mark_node(fw_node)->outputs());
    if (fw_node->get_internal_subgraphs_size() > 0) {
        auto first_body_results = fw_node->get_function(0)->get_results();
        std::vector<ResultVector> outputs;
        for (size_t i = num_skip_body_outputs; i < num_body_outs; i++) {
            outputs.push_back({first_body_results[i]});
        }
        for (size_t i = 1; i < fw_node->get_internal_subgraphs_size(); i++) {
            auto current_body_results = fw_node->get_function(i)->get_results();
            for (size_t i = num_skip_body_outputs; i < num_body_outs; i++) {
                outputs[i].push_back(current_body_results[i]);
            }
        }
        for (const auto& res_vec : outputs) {
            res.push_back(fw_node->set_body_outputs(res_vec));
        }
    }
    // Propagate extra outputs to external context
    for (const auto& output : extra_outputs_map) {
        context.add_tensor_to_context(output.first, fw_node->set_body_outputs(output.second));
    }
    return res;
}

std::shared_ptr<ov::op::util::FrameworkNode> cast_fw_node(std::shared_ptr<Node> node, const std::string& type) {
    auto fw_node = std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(node);
    if (!fw_node) {
        return nullptr;
    }
    const auto& attrs = fw_node->get_attrs();
    if (attrs.find(PtFrameworkNode::op_type_key) == attrs.end() || attrs.at(PtFrameworkNode::op_type_key) != type) {
        return nullptr;
    }
    return fw_node;
}

std::shared_ptr<ov::op::util::FrameworkNode> cast_fw_node(std::shared_ptr<Node> node,
                                                          std::initializer_list<std::string> types) {
    auto fw_node = std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(node);
    if (!fw_node) {
        return nullptr;
    }
    const auto& attrs = fw_node->get_attrs();
    for (auto type : types) {
        if (attrs.find(PtFrameworkNode::op_type_key) != attrs.end() && attrs.at(PtFrameworkNode::op_type_key) == type) {
            return fw_node;
        }
    }
    return nullptr;
}

std::shared_ptr<ov::Node> make_list_construct(const ov::OutputVector& inputs) {
    auto list_construct = std::make_shared<ov::op::util::FrameworkNode>(inputs, inputs.size());
    ov::op::util::FrameworkNodeAttrs attrs;
    attrs.set_type_name("PTFrameworkNode");
    attrs[PtFrameworkNode::op_type_key] = "prim::ListConstruct";
    list_construct->set_attrs(attrs);
    list_construct->validate_and_infer_types();
    return list_construct;
}

bool is_none_node(const Output<Node>& node) {
    if (const auto& fw_node_inp = std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(node.get_node_shared_ptr())) {
        const auto& attrs = fw_node_inp->get_attrs();
        if (attrs.find("none_value") != attrs.end()) {
            return true;
        }
    }
    return false;
}

Any simplified_type_interpret(Any type) {
    // Interpret Tensor[type] as just type
    // After applying of this interpretation we cannot distinguish true scalars (not tensors) and tensors with elements
    // of the same types
    if (type.is<type::Tensor>()) {
        const auto& tensor = type.as<type::Tensor>();
        if (tensor.element_type.is<element::Type>()) {
            return tensor.element_type;
        }
    } else if (type.is<type::PyScalar>()) {
        const auto& scalar = type.as<type::PyScalar>();
        if (scalar.element_type.is<element::Type>()) {
            return scalar.element_type;
        }
    }

    return type;
}

bool is_python_scalar_input(const NodeContext& context, size_t index) {
    return context.get_input_type(index).is<type::PyScalar>();
}

void align_eltwise_input_types(const NodeContext& context,
                               Output<Node>& lhs,
                               Output<Node>& rhs,
                               const bool& is_lhs_python_scalar,
                               const bool& is_rhs_python_scalar) {
    const auto& lhs_type = lhs.get_element_type();
    const auto& rhs_type = rhs.get_element_type();
    auto const_0 = v0::Constant::create(element::i32, Shape{}, {1});
    auto const_1 = v0::Constant::create(element::i32, Shape{1}, {1});
    // Create temporary copy of lhs and rhs for ConvertPromoteTypes to not modify original nodes.
    ov::Output<ov::Node> tmp_lhs = lhs;
    ov::Output<ov::Node> tmp_rhs = rhs;
    // Python scalar has lower priority than any tensor with any dimension.
    // If only one input is PyScalar, replace it with const to mitigate issues with dynamic type caused by dynamic
    // shape.
    if (is_lhs_python_scalar && !is_rhs_python_scalar) {
        tmp_lhs = context.mark_node(std::make_shared<v1::ConvertLike>(const_0, lhs));
        tmp_rhs = context.mark_node(std::make_shared<v1::ConvertLike>(const_1, rhs));
    } else if (!is_lhs_python_scalar && is_rhs_python_scalar) {
        tmp_lhs = context.mark_node(std::make_shared<v1::ConvertLike>(const_1, lhs));
        tmp_rhs = context.mark_node(std::make_shared<v1::ConvertLike>(const_0, rhs));
    }

    auto at = context.mark_node(std::make_shared<v14::ConvertPromoteTypes>(tmp_lhs, tmp_rhs, true, true, element::f32));
    auto dst_type = at->get_output_element_type(0);
    if (dst_type.is_dynamic()) {
        // Add ConvertLike on original node to not remove changes to shape done to differentiate between tensors and
        // scalars.
        lhs = context.mark_node(std::make_shared<v1::ConvertLike>(lhs, at->output(0)));
        rhs = context.mark_node(std::make_shared<v1::ConvertLike>(rhs, at->output(1)));
    } else {
        // Cast to destination type
        if (dst_type != lhs_type) {
            lhs = context.mark_node(std::make_shared<v0::Convert>(lhs, dst_type));
        }
        if (dst_type != rhs_type) {
            rhs = context.mark_node(std::make_shared<v0::Convert>(rhs, dst_type));
        }
    }
    return;
}

void align_output_types(const NodeContext& context, OutputVector& outputs) {
    for (size_t i = 0; i < outputs.size(); i++) {
        auto dtype_any = context.get_output_type(i);
        if (dtype_any.is<element::Type>()) {
            auto dtype = dtype_any.as<element::Type>();
            if (dtype.is_static() && dtype != outputs[i].get_element_type()) {
                outputs[i] = std::make_shared<v0::Convert>(outputs[i], dtype);
            }
        }
    }
}

Output<Node> try_constfold(const Output<Node>& x) {
    auto res = x;
    if (const auto x_const = ov::util::get_constant_from_source(x)) {
        res = x_const;
    }
    return res;
}

Output<Node> get_input_with_floating_type(const NodeContext& context, size_t idx) {
    auto x = context.get_input(static_cast<int>(idx));
    // This const only needed for type alignment
    auto dummy_const = context.mark_node(v0::Constant::create(element::f32, Shape({}), {0.5}))->output(0);
    align_eltwise_input_types(context, x, dummy_const, false, true);
    return x;
}

Output<Node> get_input_as_i32(const NodeContext& context, size_t idx) {
    auto x = context.get_input(static_cast<int>(idx));
    if (x.get_element_type() != element::i32) {
        x = context.mark_node(std::make_shared<v0::Convert>(x, element::i32));
    }
    return x;
}

Output<Node> get_input_concat_if_list(const NodeContext& context, size_t idx) {
    auto x = context.get_input(static_cast<int>(idx));
    if (context.get_input_type(idx).is<type::List>() &&
        std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(x.get_node_shared_ptr())) {
        auto elems = get_list_as_outputs(x, true);
        if (elems.size() == 0)
            // Can we figure real type for empty list?
            return std::make_shared<v0::Constant>(element::i32, Shape{0}, std::vector<int>{});
        OutputVector inputs;
        for (auto& elem : elems) {
            inputs.push_back(try_constfold(elem));
        }
        auto new_x = std::make_shared<v0::Concat>(inputs, 0);
        new_x->set_friendly_name(x.get_node_shared_ptr()->get_friendly_name());
        x = new_x;
    }
    if (const auto x_const = ov::util::get_constant_from_source(x)) {
        return x_const;
    }
    return x;
}

std::tuple<Output<Node>, Output<Node>> get_inputs_with_promoted_types(const NodeContext& context,
                                                                      size_t lhs_idx,
                                                                      size_t rhs_idx) {
    FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(lhs_idx) && !context.input_is_none(rhs_idx),
                                  "Input should not be None.");
    auto lhs = context.get_input(static_cast<int>(lhs_idx));
    auto rhs = context.get_input(static_cast<int>(rhs_idx));
    align_eltwise_input_types(context,
                              lhs,
                              rhs,
                              is_python_scalar_input(context, lhs_idx),
                              is_python_scalar_input(context, rhs_idx));
    return std::make_tuple(lhs, rhs);
}

std::deque<Output<Node>> get_list_as_outputs(const Output<Node>& start, bool unsqueeze_for_concat) {
    std::deque<Output<Node>> res;
    auto current_output = start;
    auto zero = v0::Constant::create(element::i32, Shape{}, {0});
    while (const auto& input_fw_node =
               std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(current_output.get_node_shared_ptr())) {
        const auto& attrs = input_fw_node->get_attrs();
        if (attrs.find(PtFrameworkNode::op_type_key) == attrs.end()) {
            break;
        }
        if (attrs.at(PtFrameworkNode::op_type_key) == "aten::append") {
            auto elem = input_fw_node->get_input_source_output(1);
            if (unsqueeze_for_concat) {
                elem = std::make_shared<v0::Unsqueeze>(elem, zero);
            }
            res.push_front(elem);
        } else if (attrs.at(PtFrameworkNode::op_type_key) == "aten::add") {
            const auto&& rhs_list = get_list_as_outputs(input_fw_node->get_input_source_output(1));
            res.insert(res.end(), rhs_list.begin(), rhs_list.end());
        } else {
            break;
        }
        current_output = input_fw_node->get_input_source_output(0);
    }
    auto list_construct = cast_fw_node(current_output.get_node_shared_ptr(), "prim::ListConstruct");
    if (list_construct) {
        auto inputs = list_construct->inputs();
        for (auto input_it = inputs.rbegin(); input_it != inputs.rend(); ++input_it) {
            auto elem = input_it->get_source_output();
            if (unsqueeze_for_concat) {
                elem = std::make_shared<v0::Unsqueeze>(elem, zero);
            }
            res.push_front(elem);
        }
    } else {
        res.push_front(current_output);
    }
    return res;
}

void add_exception_to_fw_node(std::shared_ptr<Node> node, const std::string& msg) {
    if (auto fw_node = ov::as_type_ptr<PtFrameworkNode>(node)) {
        auto attrs = fw_node->get_attrs();
        attrs[PtFrameworkNode::failed_conversion_key] = msg;
        fw_node->set_attrs(attrs);
    }
}

void copy_runtime_info_and_name(const std::shared_ptr<Node>& from,
                                ov::NodeVector to,
                                const ov::NodeVector& additional_rt_info_src) {
    if (to.size() == 1) {
        // We do 1 to 1 matching, no need to process names, just inherit initial name
        to[0]->set_friendly_name(from->get_friendly_name());
    } else {
        std::unordered_set<std::string> unique_names;
        size_t idx = 0;
        for (auto& op : to) {
            auto new_name = from->get_friendly_name() + '/' + op->get_type_name();
            if (unique_names.count(new_name)) {
                new_name += '_' + std::to_string(idx++);
            } else {
                unique_names.insert(new_name);
            }
            op->set_friendly_name(std::move(new_name));
        }
    }
    copy_runtime_info(from, to);
    if (!additional_rt_info_src.empty())
        copy_runtime_info(additional_rt_info_src, to);
}

// helper ops
Output<Node> masked_fill(ov::pass::NodeRegistry& rg,
                         const Output<Node>& data,
                         const Output<Node>& mask,
                         const Output<Node>& value) {
    auto _value = rg.make<v1::ConvertLike>(value, data);
    auto bool_mask = rg.make<v0::Convert>(mask, element::boolean);
    return rg.make<v1::Select>(bool_mask, _value, data);
}

Output<Node> concat_list_from_inputs(const NodeContext& context, size_t begin, size_t end) {
    OutputVector list_elems;
    for (size_t i = begin; i < end; i++) {
        if (context.get_input_type(i).as<type::List>().element_type.is<type::PyScalar>()) {
            auto const_val = context.const_input<int64_t>(i);
            std::vector<int64_t> dim_vec;
            dim_vec.push_back(const_val);
            auto dim_const = v0::Constant::create(element::i64, Shape{1}, dim_vec);
            list_elems.push_back(dim_const);
        } else {
            auto input_dim = context.get_input(static_cast<int>(i));
            if (input_dim.get_partial_shape().rank() == 0) {
                auto zero = v0::Constant::create(element::i32, Shape{}, {0});
                auto unsqueezed_dim = context.mark_node(std::make_shared<v0::Unsqueeze>(input_dim, zero));
                list_elems.push_back(unsqueezed_dim);
            } else {
                list_elems.push_back(input_dim);
            }
        }
    }
    auto concat = std::make_shared<v0::Concat>(list_elems, 0);
    return concat;
}

Output<Node> masked_select(const NodeContext& context, const Output<Node>& data, const Output<Node>& mask) {
    auto input_order = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {1, 0}));
    auto nonzero = context.mark_node(std::make_shared<v3::NonZero>(mask));
    auto masked_id = context.mark_node(std::make_shared<v1::Transpose>(nonzero, input_order));
    return context.mark_node(std::make_shared<v8::GatherND>(data, masked_id));
}

Output<Node> flatten(ov::pass::NodeRegistry& rg, const Output<Node>& value, size_t axis) {
    // First dimension of output tensor is the product of [d_0, ... d_{axis-1}] dimensions of
    // input tensor. The last dimension is the product of the rest of input tensor dimensions:
    // [d_{axis}, ..., d_n]
    Output<Node> output_shape;
    if (axis == 0) {
        output_shape = v0::Constant::create(element::i32, Shape{2}, {1, -1});
    } else if (axis == 1) {
        output_shape = v0::Constant::create(element::i32, Shape{2}, {0, -1});
    } else {
        const auto value_shape = rg.make<v3::ShapeOf>(value, element::i32);
        const auto value_rank = rg.make<v3::ShapeOf>(value_shape, element::i32);
        const auto axis_node = v0::Constant::create(element::i32, Shape{1}, {axis});
        auto start = v0::Constant::create(element::i32, Shape{1}, {0});
        auto step = v0::Constant::create(element::i32, Shape{1}, {1});
        const auto first_part_dims = rg.make<v8::Slice>(value_shape, start, axis_node, step);
        auto zero = v0::Constant::create(element::i32, {}, {0});
        auto first_part_dims_length = rg.make<v1::ReduceProd>(first_part_dims, zero, true);

        auto remaining_part_length = v0::Constant::create(element::i32, {1}, {-1});

        output_shape = rg.make<v0::Concat>(OutputVector{first_part_dims_length, remaining_part_length}, 0);
    }
    return rg.make<v1::Reshape>(value, output_shape, true);
}

bool index_tensor_on_list(ov::pass::NodeRegistry& rg,
                          const Output<Node>& data,
                          const ov::OutputVector& indices,
                          const ov::Rank& rank,
                          Output<Node>& new_output,
                          bool& use_input_as_output) {
    // Multiple tensors as indices. Each tensor could either be
    //   1. prim::Constant()
    //           representing ":" in python indexing. E.g. tensor[:, :]
    //   2. prim::Constant[value=...] or tensor output
    //           representing advanced indexing. E.g. tensor[[0, 1], [2, 0]].
    // For more info on advanced indexing,
    // check https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing

    // Consider a general case of
    //       t: [x_1, y_1, y_2, ..., x_m, ..., y_n]
    // where t is a tensor of rank m+n, {x_i} are axes where tensor index is provided, and {y_i} are axes for
    // ":". Same results can be achieved through transposing t into
    //       t: [x_1, x_2, ..., x_m, y_1, y_2, ..., y_n]
    // and use gather
    //       t: [x_1 * x_2 * ... * x_m, y_1 * y_2 * ... * y_n]
    //       tensor index = \sum_{i=1}^m (ind_i * \prod_{j=i+1}^m (x_j))
    // After gather, reshape and transpose back.
    std::vector<size_t> advanced_ids;
    std::vector<bool> is_masked_bool;
    OutputVector masked_indicies;
    // for case when index is bool e.g. x[x>0], replace index with non_zero
    for (size_t i = 0; i < indices.size(); ++i) {
        // skip dimensions where index is None
        bool is_none = false;
        if (!indices[i].get_node_shared_ptr()) {
            is_none = true;
        }
        if (auto const_input = cast_fw_node(indices[i].get_node_shared_ptr(), "prim::Constant")) {
            const auto& attrs = const_input->get_attrs();
            if (attrs.find("none_value") != attrs.end()) {
                is_none = true;
            }
        }
        if (is_none) {
            masked_indicies.push_back(indices[i]);
            is_masked_bool.push_back(false);
            continue;
        }
        auto id_dtype = indices[i].get_element_type();
        if (id_dtype == element::boolean || id_dtype == element::u8) {
            auto idx = rg.make<v0::Convert>(indices[i], element::u8);
            auto nonzero = rg.make<v3::NonZero>(idx, element::i32);
            auto input_order = rg.make<v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{1, 0});
            auto masked_id = rg.make<v1::Transpose>(nonzero, input_order);
            masked_indicies.push_back(masked_id);
            is_masked_bool.push_back(true);
        } else {
            masked_indicies.push_back(indices[i]);
            is_masked_bool.push_back(false);
        }
        advanced_ids.push_back(i);
    }

    // all indicies prim::Constant(None), return input as is
    if (advanced_ids.size() == 0) {
        new_output = data;
        use_input_as_output = true;
        return true;
    }
    // perform gather for single element case
    if (advanced_ids.size() == 1) {
        auto index = masked_indicies[advanced_ids[0]];
        if (is_masked_bool[advanced_ids[0]]) {
            auto gather = rg.make<v8::GatherND>(data, index);
            new_output = gather->output(0);
            use_input_as_output = false;
            return true;
        }
        index = rg.make<v0::Convert>(index, element::i32);
        auto dim = rg.make<v0::Constant>(element::i32, Shape{}, static_cast<int32_t>(advanced_ids[0]));
        auto gather = rg.make<v8::Gather>(data, index, dim);
        new_output = gather->output(0);
        use_input_as_output = false;
        return true;
    }
    // index transformation supports only tensors with static rank
    if (rank.is_dynamic()) {
        return false;
    }
    auto adv_idx_count = advanced_ids.size();
    auto input_shape = rg.make<v3::ShapeOf>(data, element::i32);
    auto zero = rg.make<v0::Constant>(element::i32, Shape{}, 0);
    auto input_dims = rg.make<v1::Split>(input_shape, zero, rank.get_length());
    std::vector<size_t> non_used_dims;
    for (auto i = 0; i < rank.get_length(); i++) {
        if (std::find(advanced_ids.begin(), advanced_ids.end(), i) == advanced_ids.end()) {
            non_used_dims.push_back(i);
        }
    }
    std::vector<size_t> permutation_dims;
    permutation_dims.insert(permutation_dims.end(), advanced_ids.begin(), advanced_ids.end());
    permutation_dims.insert(permutation_dims.end(), non_used_dims.begin(), non_used_dims.end());
    auto transpose_dims = rg.make<v0::Constant>(element::i32, Shape{permutation_dims.size()}, permutation_dims);
    auto transposed_input = rg.make<v1::Transpose>(data, transpose_dims);
    auto flatten_input = flatten(rg, transposed_input, adv_idx_count);
    auto cum_adv_index = masked_indicies[advanced_ids[adv_idx_count - 1]];
    cum_adv_index = rg.make<v0::Convert>(cum_adv_index, element::i32);
    auto multiplier = input_dims->output(advanced_ids[adv_idx_count - 1]);
    for (int i = static_cast<int>(adv_idx_count) - 2; i > -1; i--) {
        auto input_id = advanced_ids[i];
        auto m_idx = rg.make<v0::Convert>(masked_indicies[input_id], element::i32);
        auto adv_index = rg.make<v1::Multiply>(m_idx, multiplier);
        cum_adv_index = rg.make<v1::Add>(cum_adv_index, adv_index);
        multiplier = rg.make<v1::Multiply>(multiplier, input_dims->output(input_id));
    }
    std::shared_ptr<Node> gather = rg.make<v8::Gather>(flatten_input, cum_adv_index, zero);
    OutputVector concat_dims;
    // check if all advanced indices are consecutive.
    std::vector<size_t> consequence_dims;
    auto cum_adv_index_shape_tensor = rg.make<v3::ShapeOf>(cum_adv_index, element::i32);
    for (size_t i = advanced_ids[0]; i <= advanced_ids[advanced_ids.size() - 1]; i++) {
        consequence_dims.push_back(i);
    }
    // unfold regular index axes
    if (advanced_ids == consequence_dims) {
        OutputVector folded_adv_idx_shape_vector;
        auto minus_one = rg.make<v0::Constant>(element::i32, Shape{1}, -1);
        folded_adv_idx_shape_vector.push_back(minus_one);
        for (auto i : non_used_dims) {
            folded_adv_idx_shape_vector.push_back(input_dims->output(i));
        }
        auto folded_adv_idx_shape = rg.make<v0::Concat>(folded_adv_idx_shape_vector, 0);
        gather = rg.make<v1::Reshape>(gather, folded_adv_idx_shape, false);
        std::vector<size_t> adv_idx_permute;
        for (size_t i = 1; i < advanced_ids[0] + 1; i++) {
            adv_idx_permute.push_back(i);
        }
        adv_idx_permute.push_back(0);
        for (size_t i = advanced_ids[0] + 1; i < (rank.get_length() - adv_idx_count + 1); i++) {
            adv_idx_permute.push_back(i);
        }
        // Transpose folded advanced indexed axis to its original location.
        auto permute_indicies = rg.make<v0::Constant>(element::i32, Shape{adv_idx_permute.size()}, adv_idx_permute);
        gather = rg.make<v1::Transpose>(gather, permute_indicies);
        // unfold advanced index axes
        for (size_t i = 0; i < advanced_ids[0]; i++) {
            concat_dims.push_back(input_dims->output(i));
        }
        concat_dims.push_back(cum_adv_index_shape_tensor);
        for (auto i : non_used_dims) {
            if (i < advanced_ids[0]) {
                continue;
            }
            concat_dims.push_back(input_dims->output(i));
        }

    } else {
        concat_dims.push_back(cum_adv_index_shape_tensor);
        for (auto i : non_used_dims) {
            concat_dims.push_back(input_dims->output(i));
        }
    }
    auto final_shape = rg.make<v0::Concat>(concat_dims, 0);
    gather = rg.make<v1::Reshape>(gather, final_shape, false);
    new_output = gather->output(0);
    use_input_as_output = false;
    return true;
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
