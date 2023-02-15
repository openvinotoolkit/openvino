// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "op_table.hpp"
#include "openvino/frontend/pytorch/decoder.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/util/log.hpp"
#include "pt_framework_node.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs) {
    auto inputs = context.inputs();
    FRONT_END_OP_CONVERSION_CHECK(inputs.size() >= min_inputs, "Got less inputs than expected");
    for (auto i = max_inputs; i < inputs.size(); i++) {
        FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(i), "Got more inputs than expected.");
    }
}

Output<Node> make_optional_bias(const Output<Node>& base_op,
                                const NodeContext& context,
                                int bias_input_idx,
                                const std::vector<int>& unsqueeze_dims) {
    using std::make_shared;

    if (!context.input_is_none(bias_input_idx)) {
        auto bias = context.get_input(bias_input_idx);
        if (!unsqueeze_dims.empty()) {
            auto indices = opset10::Constant::create(element::i32, {unsqueeze_dims.size()}, unsqueeze_dims);
            context.mark_node(indices);
            bias = make_shared<opset10::Unsqueeze>(bias, indices);
            context.mark_output(bias);
        }
        return make_shared<opset10::Add>(context.mark_output(base_op), bias);
    } else {
        return base_op;
    }
}

Output<Node> reshape_channelwise(const NodeContext& context,
                                 const Output<Node>& data,
                                 const Output<Node>& shape_source) {
    auto input_shape = context.mark_node(std::make_shared<opset10::ShapeOf>(shape_source));
    auto input_rank = context.mark_node(std::make_shared<opset10::ShapeOf>(input_shape));
    auto one_const = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {1}));
    auto two_const = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {2}));
    auto tail_shape_rank = context.mark_node(std::make_shared<opset10::Subtract>(input_rank, two_const));
    auto tail_shape = context.mark_node(std::make_shared<opset10::Broadcast>(one_const, tail_shape_rank));
    auto channels_dim = context.mark_node(std::make_shared<opset10::ShapeOf>(data));
    auto new_shape =
        context.mark_node(std::make_shared<opset10::Concat>(OutputVector{one_const, channels_dim, tail_shape}, 0));

    return context.mark_node(std::make_shared<opset10::Reshape>(data, new_shape, false));
}

std::tuple<Output<Node>, Output<Node>> get_shape_rank(const NodeContext& context,
                                                      const Output<Node>& x,
                                                      bool as_scalar,
                                                      element::Type output_type) {
    auto shape = context.mark_node(std::make_shared<opset10::ShapeOf>(x, output_type));
    Output<Node> rank = context.mark_node(std::make_shared<opset10::ShapeOf>(shape, output_type));
    if (as_scalar) {
        rank = context.mark_node(std::make_shared<opset10::Squeeze>(rank));
    }
    return std::make_tuple(shape, rank);
}

Output<Node> reshape_kernel_for_group(const NodeContext& context, const Output<Node>& kernel, int64_t groups) {
    using std::make_shared;

    auto axis_0 = opset10::Constant::create(element::i64, Shape{}, {0});
    auto groups_const = opset10::Constant::create(element::i64, Shape{1}, {groups});
    auto neg_1_const = opset10::Constant::create(element::i64, Shape{1}, {-1});

    auto kernel_shape = std::make_shared<opset10::ShapeOf>(kernel);
    auto c_out_idx = opset10::Constant::create(element::i64, Shape{}, {0});
    auto kernel_shape_0 = make_shared<opset10::Gather>(kernel_shape, c_out_idx, axis_0);
    auto kernel_shape_0_uns = make_shared<opset10::Unsqueeze>(kernel_shape_0, axis_0);
    auto c_out_value = make_shared<opset10::Divide>(kernel_shape_0_uns, groups_const);

    auto start = opset10::Constant::create(element::i64, Shape{1}, {2});
    auto stop = opset10::Constant::create(element::i64, Shape{1}, {std::numeric_limits<int64_t>::max()});
    auto step = opset10::Constant::create(element::i64, Shape{1}, {1});
    auto remaining_shape = make_shared<opset10::Slice>(kernel_shape, start, stop, step);

    auto new_kernel_shape =
        make_shared<opset10::Concat>(OutputVector{groups_const, c_out_value, neg_1_const, remaining_shape}, 0);
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
                        new_kernel_shape});
    return make_shared<opset10::Reshape>(kernel, new_kernel_shape, false);
}

std::shared_ptr<Node> get_axes_range(const NodeContext& context, int input_id) {
    auto x = context.get_input(input_id);
    auto start = std::make_shared<opset10::Constant>(element::i32, Shape{}, 0);
    auto step = std::make_shared<opset10::Constant>(element::i32, Shape{}, 1);
    auto shape = context.mark_node(std::make_shared<opset10::ShapeOf>(x, element::i32));
    auto rank = context.mark_node(std::make_shared<opset10::ShapeOf>(shape, element::i32));
    auto reduced_rank = context.mark_node(std::make_shared<opset10::Squeeze>(rank));
    return context.mark_node(std::make_shared<opset10::Range>(start, reduced_rank, step, element::i32));
};

std::shared_ptr<Node> numel(const NodeContext& context, const Output<Node>& x) {
    auto input_shape = context.mark_node(std::make_shared<opset10::ShapeOf>(x));
    auto axes = context.mark_node(opset10::Constant::create(element::i64, Shape({1}), {0}));
    return context.mark_node(std::make_shared<opset10::ReduceProd>(input_shape, axes, false));
};

namespace {
const std::unordered_map<int64_t, element::Type> TORCH_TO_OV_TYPE{{0, element::u8},
                                                                  {1, element::i8},
                                                                  {2, element::i16},
                                                                  {3, element::i32},
                                                                  {4, element::i64},
                                                                  {5, element::f16},
                                                                  {6, element::f32},
                                                                  {7, element::f64},
                                                                  {11, element::boolean}};

const std::unordered_map<std::string, ov::op::PadType> TORCH_AUTO_PAD_TO_OV{{"valid", ov::op::PadType::VALID},
                                                                            {"same", ov::op::PadType::SAME_UPPER}};
}  // namespace

element::Type convert_dtype(int64_t pt_type) {
    FRONT_END_OP_CONVERSION_CHECK(TORCH_TO_OV_TYPE.count(pt_type), "Unknown type: ", pt_type);
    return TORCH_TO_OV_TYPE.at(pt_type);
};

ov::op::PadType convert_pad(const std::string& pt_pad) {
    FRONT_END_OP_CONVERSION_CHECK(TORCH_AUTO_PAD_TO_OV.count(pt_pad), "Unknown pad: ", pt_pad);
    return TORCH_AUTO_PAD_TO_OV.at(pt_pad);
};

std::shared_ptr<Node> concat_list_construct(std::shared_ptr<Node> input) {
    if (auto list_construct = cast_fw_node(input, "prim::ListConstruct")) {
        auto list_inputs = list_construct->input_values();
        OutputVector node_vector;
        auto zero = opset10::Constant::create(element::i32, Shape{}, {0});
        for (size_t i = 0; i < list_inputs.size(); i++) {
            auto node = concat_list_construct(list_inputs[i].get_node_shared_ptr());
            auto unsqueezed_node = std::make_shared<opset10::Unsqueeze>(node, zero);
            node_vector.push_back(unsqueezed_node);
        }
        return std::make_shared<opset10::Concat>(node_vector, 0);
    }
    return input;
}

OutputVector make_framework_node(NodeContext* context) {
    auto schema = context->get_schema();
    // TODO: properly process schema to get the actual position of mutable input
    // Hack. Can indicate mutable inputs, but can it be reliable?
    if (schema.find('!') != std::string::npos) {
        // We create additional output for such nodes. It contains new tensor that represents input that was changed.
        auto fw_node = std::make_shared<PtFrameworkNode>(context->get_decoder(),
                                                         context->inputs(),
                                                         context->get_output_size() + 1);
        fw_node->set_friendly_name(context->get_op_type());
        auto outputs = fw_node->outputs();
        // Usually mutated input index is 0, because it is usually "self" input, so we need to replace this tensor with
        // output we created.
        context->mutate_input(0, outputs.back());
        OPENVINO_DEBUG << "Created node with mutated 0 input. Schema: " << schema << '\n';
        context->mark_node(fw_node);
        // For simplification we do not expect such operations to have extra bodies
        FRONT_END_OP_CONVERSION_CHECK(context->get_decoder()->get_subgraph_size() == 0,
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
    for (size_t i = 0; i < context->get_decoder()->get_subgraph_size(); ++i) {
        auto subgraph_decoder = context->get_decoder()->get_subgraph_decoder(i);
        auto inputs = subgraph_decoder->inputs();
        input_idxs.insert(inputs.begin(), inputs.end());
        auto body = context->convert_subgraph(i);
        bodies.push_back(body);
        for (const auto& param : body->get_parameters()) {
            auto name = param->get_output_tensor(0).get_any_name();
            size_t input_idx = (size_t)std::stoll(name);
            inputs_map[input_idx].push_back(param);
        }
        auto body_outputs = subgraph_decoder->outputs();
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
            auto name = body_results[i]->input(0).get_tensor().get_any_name();
            size_t out_idx = (size_t)std::stoll(name);
            FRONT_END_OP_CONVERSION_CHECK(extra_outputs_map.count(out_idx) == 0,
                                          "More then one body output with same tensor name.");
            extra_outputs_map[out_idx].push_back(body_results[i]);
        }
    }
    // Number of body outputs can be higher then number of pt node outputs, e.g. in case of loop first body output is
    // condition, we have to skip such outputs.
    auto num_skip_body_outputs =
        num_body_outs > context->get_output_size() ? num_body_outs - context->get_output_size() : 0;

    // We need to reduce number of outputs, because some outputs are outputs from body
    auto fw_node =
        std::make_shared<PtFrameworkNode>(context->get_decoder(),
                                          context->inputs(),
                                          context->get_output_size() - num_body_outs + num_skip_body_outputs);
    fw_node->set_friendly_name(context->get_op_type());
    for (size_t i = 0; i < bodies.size(); ++i) {
        fw_node->set_function(static_cast<int>(i), bodies[i]);
    }

    // Connect inputs with external context
    for (const auto& input : inputs_map) {
        if (!input_idxs.count(input.first)) {
            auto external_output = context->get_tensor_from_model_or_create_input(input.first);
            fw_node->set_invariant_inputs(external_output, input.second);
        } else {
            auto external_output = context->get_tensor_from_model(input.first);
            if (external_output.get_node()) {
                fw_node->set_invariant_inputs(external_output, input.second);
            }
        }
    }
    OutputVector res(context->mark_node(fw_node)->outputs());
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
        context->add_tensor_to_context(output.first, fw_node->set_body_outputs(output.second));
    }
    return res;
}

OutputVector convert_node(NodeContext* context) {
    try {
        auto CONVERTERS_MAP = get_supported_ops();
        auto it = CONVERTERS_MAP.find(context->get_op_type());
        if (it != CONVERTERS_MAP.end()) {
            return it->second(*context);
        }

    } catch (std::runtime_error& e) {
        OPENVINO_DEBUG << "Exception happened during conversion of op: " << context->get_op_type()
                       << " with schema: " << context->get_schema() << ": " << e.what() << '\n';
    } catch (...) {
        OPENVINO_DEBUG << "Some exception happened during conversion of node of type: " << context->get_op_type()
                       << '\n';
    }
    // Create PtFrameworkNode for everything that wasn't able to be converted normally
    return make_framework_node(context);
}

/// \brief Completely convert pytorch_model, creates PtFrameworkNode if not possible to convert node
/// \param pytorch_model Input model
/// \param external_tensor_map Is used for recursive calls of convert_pytorch_model and represent the external context
///  which is visible from nested model. Empty external_tensor_map is used as an indication that this is a main body
///  conversion.
/// \return fully converted OV Model
std::shared_ptr<Model> convert_pytorch_model(std::shared_ptr<TorchDecoder> pytorch_model,
                                             const TensorMap& external_tensor_map) {
    std::shared_ptr<Model> resulting_model;  // define here to make a conversion in a nested scope
    {
        ParameterVector parameters;
        TensorMap tensor_map;  // tensor map of the current context
        std::set<size_t> mutated_tensors;

        //  Go over all pytorch_model inputs and register them in the tensor map:
        auto inputs = pytorch_model->inputs();
        for (size_t i = 0; i < inputs.size(); ++i) {
            PartialShape ps = pytorch_model->get_input_shape(i);
            auto type = simplified_type_interpret(pytorch_model->get_input_type(i));
            // TODO: Use special API to set custom type detalization
            auto parameter = std::make_shared<opset10::Parameter>(element::dynamic, ps);
            parameter->get_output_tensor(0).add_names({std::to_string(inputs.at(i))});
            parameters.push_back(parameter);
            auto order = pytorch_model->get_input_transpose_order(i);
            if (order.size() > 0 && !std::is_sorted(order.begin(), order.end())) {
                FRONT_END_GENERAL_CHECK(ps.is_static(), "Shape must be static.");  // TODO: make dynamic
                auto sh = ps.get_shape();
                Shape new_shape(sh.size());
                for (size_t i = 0; i < sh.size(); i++) {
                    new_shape[order[i]] = sh[i];
                }
                auto shape_const = opset10::Constant::create(element::i64, {new_shape.size()}, new_shape);
                auto reshape = std::make_shared<opset10::Reshape>(parameter, shape_const, false);
                auto order_const = opset10::Constant::create(element::i32, {order.size()}, order);
                auto transpose = std::make_shared<opset10::Transpose>(reshape, order_const);
                tensor_map[inputs.at(i)] = transpose;
            } else {
                tensor_map[inputs.at(i)] = parameter;
            }
        }

        auto node_visitor = [&](std::shared_ptr<TorchDecoder> node) {
            // Explore all inputs of node. Node may refer to input value that hasn't been created in the current scope.
            // But this value can be found in the outer scope, for this purpose we create new input for the model to
            // link with external scope on a higher level.

            auto raw_inputs = node->inputs();
            for (size_t i = 0; i < raw_inputs.size(); ++i) {
                auto input = raw_inputs.at(i);
                if (tensor_map.find(input) == tensor_map.end()) {
                    // Input refers value in the outer scope, need to create a new Parameter in the current scope
                    // Linkage to external scope will be performed on the level of the parent operation (if or loop)
                    // TODO: Eliminate duplication with the main code for Parameters creation
                    PartialShape ps = node->get_input_shape(i);
                    auto type = simplified_type_interpret(node->get_input_type(i));
                    // TODO: Use special API to set custom type detalization
                    auto parameter = std::make_shared<opset10::Parameter>(element::dynamic, ps);
                    // TODO: Missing get_input_transpose_order handling for not trivial layouts
                    tensor_map[input] = parameter;
                    // set name of parameter to the index of node in the model
                    parameter->get_output_tensor(0).add_names({std::to_string(input)});
                    parameters.push_back(parameter);
                }
            }
            auto context = NodeContext(node, &tensor_map, &parameters, external_tensor_map);
            auto converted_outputs = convert_node(&context);

            auto mutated_t = context.get_mutated_tensors();
            mutated_tensors.insert(mutated_t.begin(), mutated_t.end());

            auto fw_outputs = node->outputs();
            // Ops with subgraphs or with mutated inputs may have more outputs after conversion compared to pytorch ones
            FRONT_END_OP_CONVERSION_CHECK(fw_outputs.size() <= converted_outputs.size(),
                                          "Number of ",
                                          node->get_op_type(),
                                          " outputs greater then number of converted outputs.");

            // TODO: Make sure that mapping of fw_outputs to converted_outputs does always work
            // FIXME: Now it is not true for at least prim::Constant
            for (size_t i = 0; i < fw_outputs.size(); ++i) {
                size_t fw_tensor_id = node->output(i);
                if (tensor_map.find(fw_tensor_id) != tensor_map.end()) {
                    throw std::runtime_error("Duplicated producer for PT value with unique ID: " +
                                             std::to_string(fw_tensor_id));
                }

                // Output shape of converted node should match the original output shape
                // OV_FRONTEND_REQUIRE(get_ov_shape(fw_outputs[i]) == converted_outputs[i].get_partial_shape());

                tensor_map[fw_tensor_id] = converted_outputs[i];
                converted_outputs[i].get_tensor().add_names({std::to_string(fw_tensor_id)});
            }
        };

        FRONT_END_GENERAL_CHECK(pytorch_model->get_subgraph_size() == 1, "Model should have exactly 1 subgraph.");
        pytorch_model->visit_subgraph(node_visitor);

        ResultVector results;
        for (size_t i = 0; i < pytorch_model->num_of_outputs(); ++i) {
            size_t id = pytorch_model->output(i);
            if (tensor_map.find(id) == tensor_map.end()) {
                // Not found in this scope, adding Parameter to connect to external scope
                auto parameter = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
                parameter->get_output_tensor(0).add_names({std::to_string(id)});
                parameters.push_back(parameter);
                tensor_map[id] = parameter;
            }
            auto ov_output = tensor_map[id];
            auto order = pytorch_model->get_output_transpose_order(i);
            FRONT_END_GENERAL_CHECK(order.size() == 0 || std::is_sorted(order.begin(), order.end()),
                                    "Output strides have wrong order.");
            FRONT_END_GENERAL_CHECK(ov_output.get_names().size() > 0,
                                    "Tensor doesn't have name, while it should have name: ",
                                    id);
            auto result = std::make_shared<opset10::Result>(ov_output);
            results.push_back(result);
        }

        // Since parameters can be added we need to list all current parameters
        std::set<size_t> param_names;
        for (const auto& param : parameters) {
            auto name = param->get_output_tensor(0).get_any_name();
            size_t input_idx = (size_t)std::stoll(name);
            param_names.insert(input_idx);
        }
        for (const auto& tensor_id : mutated_tensors) {
            if (param_names.count(tensor_id)) {
                FRONT_END_GENERAL_CHECK(tensor_map.count(tensor_id),
                                        "Tensor with id: ",
                                        tensor_id,
                                        " doesn't exist in tensor map.");
                // model input was mutated we need to make a result for it
                auto mutated_tensor = tensor_map.at(tensor_id);
                // empty external_tensor_map means this is main body of the model and we don't want to create
                // additional outputs in that case.
                if (mutated_tensor.get_target_inputs().empty() && !external_tensor_map.empty())
                    results.push_back(std::make_shared<opset10::Result>(tensor_map.at(tensor_id)));
            }
        }
        resulting_model = std::make_shared<Model>(results, parameters);
        // Did a conversion in a nested scope to automatically remove any holders of nodes except those in the graph
    }

    return resulting_model;
}

std::shared_ptr<ov::op::util::FrameworkNode> cast_fw_node(std::shared_ptr<Node> node, const std::string& type) {
    auto fw_node = std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(node);
    if (!fw_node) {
        return nullptr;
    }
    const auto& attrs = fw_node->get_attrs();
    if (attrs.find("PtTypeName") == attrs.end() || attrs.at("PtTypeName") != type) {
        return nullptr;
    }
    return fw_node;
}

Any simplified_type_interpret(Any type) {
    // Interpret Tensor[type] as just type
    // After applying of this interpretation we cannot distinguish true scalars (not tensors) and tensors with elements
    // of the same types
    if (type.is<type::Tensor>()) {
        auto tensor = type.as<type::Tensor>();
        if (tensor.element_type.is<element::Type>()) {
            return tensor.element_type;
        }
    }

    return type;
}

namespace {
std::unordered_map<size_t, element::Type> bit_to_float{
    {16, element::f16},
    {32, element::f32},
    {64, element::f64},
};
std::unordered_map<size_t, element::Type> bit_to_int{
    // {4, element::i4}, torch don't have int4
    {8, element::i8},
    {16, element::i16},
    {32, element::i32},
    {64, element::i64},
};
}  // namespace

void align_eltwise_input_types(const NodeContext& context, Output<Node>& lhs, Output<Node>& rhs, bool align_scalars) {
    const auto& lhs_type = lhs.get_element_type();
    const auto& rhs_type = rhs.get_element_type();
    if (lhs_type.is_dynamic() || rhs_type.is_dynamic()) {
        // if any of types is not known, align to lhs type.
        // TODO: can be fixed with special operation?
        rhs = context.mark_node(std::make_shared<opset10::ConvertLike>(rhs, lhs));
        return;
    }

    // Both types are static, align types. If float and int types are used convert int type to f32, after that align
    // to the largest bitness, if both float or both int, just align bitness
    if (lhs_type == rhs_type)
        return;

    // if one of operands is scalar, the resulting type is taken from the other operand except when scalar is float
    // type and other operand is int, in that case BOTH operands get fp32 type
    const auto& lhs_rank = lhs.get_partial_shape().rank();
    const auto& rhs_rank = rhs.get_partial_shape().rank();
    // consider dynamic rank as non scalar
    const auto is_lhs_scalar = lhs_rank.is_static() && lhs_rank.get_length() == 0;
    const auto is_rhs_scalar = rhs_rank.is_static() && rhs_rank.get_length() == 0;
    if (is_lhs_scalar && is_rhs_scalar) {
        // if both scalar, align to lhs
        rhs = context.mark_node(std::make_shared<opset10::ConvertLike>(rhs, lhs));
        return;
    }
    auto lhs_dst_type = lhs_type;
    auto rhs_dst_type = rhs_type;
    if (is_lhs_scalar) {
        if (lhs_type.is_real() && !rhs_type.is_real()) {
            // if div we need to also align float types to highest bitness regardless of scalar
            if (!align_scalars)
                lhs_dst_type = element::f32;
            rhs_dst_type = element::f32;
        } else {
            lhs = context.mark_node(std::make_shared<opset10::ConvertLike>(lhs, rhs));
            return;
        }
    } else if (is_rhs_scalar) {
        if (!lhs_type.is_real() && rhs_type.is_real()) {
            lhs_dst_type = element::f32;
            // if div we need to also align float types to highest bitness regardless of scalar
            if (!align_scalars)
                rhs_dst_type = element::f32;
        } else {
            rhs = context.mark_node(std::make_shared<opset10::ConvertLike>(rhs, lhs));
            return;
        }
    }

    if (lhs_dst_type == element::boolean || rhs_dst_type == element::boolean) {
        // Do nothing with bool
        return;
    }

    if (!lhs_dst_type.is_real() && rhs_dst_type.is_real()) {
        lhs_dst_type = element::f32;
    } else if (lhs_dst_type.is_real() && !rhs_dst_type.is_real()) {
        rhs_dst_type = element::f32;
    }
    // Align bitness to higher
    if (lhs_dst_type.bitwidth() != rhs_dst_type.bitwidth()) {
        const auto dst_bitness = std::max(lhs_dst_type.bitwidth(), rhs_dst_type.bitwidth());
        element::Type* type_to_align = &lhs_dst_type;
        if (rhs_dst_type.bitwidth() < dst_bitness)
            type_to_align = &rhs_dst_type;
        if (type_to_align->is_real()) {
            *type_to_align = bit_to_float.at(dst_bitness);
        } else {
            *type_to_align = bit_to_int.at(dst_bitness);
        }
    }

    // Cast to destination types
    if (lhs_dst_type != lhs_type) {
        lhs = context.mark_node(std::make_shared<opset10::Convert>(lhs, lhs_dst_type));
    }
    if (rhs_dst_type != rhs_type) {
        rhs = context.mark_node(std::make_shared<opset10::Convert>(rhs, rhs_dst_type));
    }
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
