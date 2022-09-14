#include "utils.hpp"

#include <openvino/frontend/pytorch/decoder.hpp>
#include <openvino/frontend/pytorch/node_context.hpp>

#include "exception.hpp"
#include "op_table.hpp"
#include "pt_framework_node.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
int LEVEL = 0;
int NUMBER = 0;
int COUNTER = 0;

Output<Node> make_optional_bias(Output<Node> base_op,
                                const NodeContext& context,
                                size_t bias_input_idx,
                                std::vector<int> unsqueeze_dims) {
    using namespace ngraph;
    using std::make_shared;

    if (!context.input_is_none(bias_input_idx)) {
        auto bias = context.get_input(bias_input_idx);
        if (!unsqueeze_dims.empty()) {
            auto indices = opset8::Constant::create(element::i32, {unsqueeze_dims.size()}, unsqueeze_dims);
            context.mark_node(indices);
            bias = make_shared<opset8::Unsqueeze>(bias, indices);
            context.mark_output(bias);
        }
        return make_shared<opset8::Add>(context.mark_output(base_op), bias);
    } else {
        return base_op;
    }
}

std::shared_ptr<ov::Node> get_rank_node(ov::Output<ov::Node> node) {
    auto shape = std::make_shared<opset8::ShapeOf>(node);
    return std::make_shared<opset8::ShapeOf>(shape);
}

Output<Node> reshape_kernel_for_group(const NodeContext& context,
                                      Output<Node> input,
                                      Output<Node> kernel,
                                      int64_t groups) {
    using namespace ngraph;
    using std::make_shared;

    auto in_shape = std::make_shared<opset8::ShapeOf>(input);
    auto c_in_idx = opset8::Constant::create(element::i64, Shape{}, {1});
    auto axis_0 = opset8::Constant::create(element::i64, Shape{}, {0});
    auto in_shape_1 = make_shared<opset8::Gather>(in_shape, c_in_idx, axis_0);
    auto in_shape_1_uns = make_shared<opset8::Unsqueeze>(in_shape_1, axis_0);
    auto groups_const = opset8::Constant::create(element::i64, Shape{1}, {groups});
    auto c_in_value = make_shared<opset8::Divide>(in_shape_1_uns, groups_const);

    auto kernel_shape = std::make_shared<opset8::ShapeOf>(kernel);
    auto c_out_idx = opset8::Constant::create(element::i64, Shape{}, {0});
    auto kernel_shape_0 = make_shared<opset8::Gather>(kernel_shape, c_out_idx, axis_0);
    auto kernel_shape_0_uns = make_shared<opset8::Unsqueeze>(kernel_shape_0, axis_0);
    auto c_out_value = make_shared<opset8::Divide>(kernel_shape_0_uns, groups_const);

    auto start = opset8::Constant::create(element::i64, Shape{1}, {2});
    auto stop = opset8::Constant::create(element::i64, Shape{1}, {std::numeric_limits<int64_t>::max()});
    auto step = opset8::Constant::create(element::i64, Shape{1}, {1});
    auto remaining_shape = make_shared<opset8::Slice>(kernel_shape, start, stop, step);

    auto new_kernel_shape =
        make_shared<opset8::Concat>(OutputVector{groups_const, c_out_value, c_in_value, remaining_shape}, 0);
    context.mark_nodes({in_shape,
                        c_in_idx,
                        axis_0,
                        in_shape_1,
                        in_shape_1_uns,
                        groups_const,
                        c_in_value,
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
    return make_shared<opset8::Reshape>(kernel, new_kernel_shape, false);
}

OutputVector convert_node(NodeContext* context) {
    // std::cout << "[  ----  DEBUG  ---- ] convert_node\n";

    // std::cerr << "---\nAttempting to convert " << node->kind().toQualString() << "\n";
    // node->dump();

    // std::cerr << "[ DEBUG ] Attempting to convert " << context.get_op_type() << "\n";

    try {
        auto CONVERTERS_MAP = get_supported_ops();
        auto it = CONVERTERS_MAP.find(context->get_op_type());
        if (it != CONVERTERS_MAP.end()) {
            // std::cout << "FOUND converter for " << context.get_op_type() << "\n";
            return it->second(*context);
        } else {
            const std::set<std::string> known_skips{"prim::RaiseException",
                                                    "aten::warn",
                                                    "prim::TupleConstruct",
                                                    "prim::ListConstruct",
                                                    "aten::format",
                                                    "aten::append",
                                                    "aten::update",
                                                    "aten::dict",
                                                    "aten::list",
                                                    "aten::_set_item",
                                                    "aten::__getitem__",
                                                    "aten::__isnot__",
                                                    "aten::__contains__",
                                                    "prim::unchecked_cast",
                                                    "prim::Uninitialized",
                                                    "prim::SetAttr",
                                                    "prim::GetAttr",
                                                    "prim::ListUnpack",
                                                    "aten::__not__"};
            if (!known_skips.count(context->get_op_type())) {
                std::cout << "DIDN'T FIND converter for " << context->get_op_type() << " with inputs:";
                if (context->inputs().size() == 0) {
                    std::cout << " None";
                }
                for (auto input : context->inputs()) {
                    std::cout << " " << input;
                }
                std::cout << " with schema: " << context->get_schema() << std::endl;
            }
        }

    }
    // catch(pybind11::error_already_set& e) {
    //     std::cout << "Python exception: " << e << "\n";
    // }
    catch (std::runtime_error& e) {
        std::cout << "Exception happened during conversion of op: " << context->get_op_type() << ": " << e.what()
                  << '\n';
        // throw;
    } catch (...) {
        std::cout << "Some exception happened during conversion of node of type: " << context->get_op_type()
                  << std::endl;
        // throw;
    }
    // if (node->kind() != prim::ListConstruct) {
    //     std::cout << "Making unsupported " << node->kind().toQualString() << std::endl;
    //     node->dump();
    // }

    // Create PtFrameworkNode for everything that wasn't able to be converted normally
    // Pay attention to subgraphs that may appear in the node
    // std::cerr << "[ DEBUG ] Before PtFramewokNode creation\n";

    auto schema = context->get_schema();
    if (schema.find('!') != std::string::npos) {
        // Hack. Can indicate mutable inputs, but can it be reliable?
        auto fw_node = std::make_shared<PtFrameworkNode>(context->get_decoder(),
                                                         context->inputs(),
                                                         context->get_decoder()->num_of_outputs() + 1);
        fw_node->set_friendly_name(context->get_op_type() + ":" + std::to_string(COUNTER++));
        auto outputs = fw_node->outputs();
        // update writes to input 0, so we need to replace this input with output from update
        context->mutate_input(0, outputs.back());
        std::cerr << "[ WARNING ] Created node with mutated 0 input. Schema: " << schema << std::endl;
        context->get_decoder()->mark_node(fw_node);
        return outputs;
    }
    auto fw_node = std::make_shared<PtFrameworkNode>(context->get_decoder(),
                                                     context->inputs(),
                                                     context->get_decoder()->num_of_outputs());
    fw_node->set_friendly_name(context->get_op_type() + ":" + std::to_string(COUNTER++));

    std::map<size_t, ParameterVector> inputs_map;
    std::map<size_t, ResultVector> outputs_map;
    std::set<size_t> input_idxs;
    for (size_t i = 0; i < context->get_decoder()->get_subgraph_size(); ++i) {
        auto subgraph_decoder = context->get_decoder()->get_subgraph_decoder(i);
        auto inputs = subgraph_decoder->inputs();
        input_idxs.insert(inputs.begin(), inputs.end());
        auto body = convert_pytorch_model(subgraph_decoder);
        fw_node->set_function(i, body);
        for (auto param : body->get_parameters()) {
            auto name = param->get_output_tensor(0).get_any_name();
            size_t input_idx = (size_t)std::stoll(name);
            inputs_map[input_idx].push_back(param);
        }
        for (auto result : body->get_results()) {
            auto name = result->input(0).get_tensor().get_any_name();
            size_t out_idx = (size_t)std::stoll(name);
            FRONT_END_OP_CONVERSION_CHECK(outputs_map.count(out_idx) == 0,
                                          "More then one body output with same tensor name.");
            outputs_map[out_idx].push_back(result);
        }
    }
    for (auto input : inputs_map) {
        if (!input_idxs.count(input.first)) {
            auto external_output = context->get_tensor_from_ext_or_create_ext_input(input.first);
            fw_node->set_invariant_inputs(external_output, input.second);
        } else {
            auto external_output = context->get_tensor_from_ext(input.first);
            if (external_output.get_node()) {
                fw_node->set_invariant_inputs(external_output, input.second);
            }
        }
    }
    for (auto output : outputs_map) {
        context->add_tensor_to_external_context(output.first, fw_node->set_body_outputs(output.second));
    }
    return context->get_decoder()->mark_node(fw_node)->outputs();
}

std::shared_ptr<ov::Model> convert_pytorch_model(std::shared_ptr<Decoder> pytorch_model) {
    LEVEL++;
    // std::cout << "=====Convert model:" << LEVEL << " start=====" << std::endl;
    std::shared_ptr<ov::Model> resulting_model;  // define here to make a conversion in a nested scope
    {
        ParameterVector parameters;
        TensorMap tensor_map;  // tensor map of the current context
        std::set<size_t> mutated_tensors;

        //  Go over all pytorch_model inputs and register them in the tensor map:
        auto inputs = pytorch_model->inputs();
        // std::cout << "[  ---  DEBUG --- ] convert_pytorch_model: number of inputs: " << inputs.size() << '\n';
        for (int i = 0; i < inputs.size(); ++i) {
            // std::cout << "Input: " << i << ": " << inputs[i] << "\n";
            PartialShape ps = pytorch_model->get_input_shape(i);
            // std::cout << "PartialShape = " << ps << "\n";
            auto parameter =
                std::make_shared<opset8::Parameter>(ov::element::custom, pytorch_model->get_input_type(i), ps);
            parameter->get_output_tensor(0).add_names({std::to_string(pytorch_model->input(i))});
            // std::cout << "Parameter: " << parameter << "\n";
            parameters.push_back(parameter);
            auto order = pytorch_model->get_input_transpose_order(i);
            if (order.size() > 0 && !std::is_sorted(order.begin(), order.end())) {
                OV_FRONTEND_REQUIRE(ps.is_static());  // TODO: make dynamic
                auto sh = ps.get_shape();
                Shape new_shape(sh.size());
                for (int i = 0; i < sh.size(); i++) {
                    new_shape[order[i]] = sh[i];
                }
                auto shape_const = opset8::Constant::create(element::i64, {new_shape.size()}, new_shape);
                auto reshape = std::make_shared<opset8::Reshape>(parameter, shape_const, false);
                auto order_const = opset8::Constant::create(element::i32, {order.size()}, order);
                auto transpose = std::make_shared<opset8::Transpose>(reshape, order_const);
                tensor_map[pytorch_model->input(i)] = transpose;
            } else {
                tensor_map[pytorch_model->input(i)] = parameter;
            }
            // std::cout << "Level:" << LEVEL << " Added model input: " << tensor_map[pytorch_model->input(i)] <<
            // std::endl;
        }

        auto node_visitor = [&](std::shared_ptr<Decoder> node) {
            // std::cerr << "Node convert start" << std::endl;

            // Even if node has no outputs it can mutate input. Remove this
            /*if (!node->num_of_outputs()) {
                bool no_subgraph_outputs = true;
                for (int i = 0; i < node->get_subgraph_size(); i++) {
                    auto subgraph = node->get_subgraph_decoder(i);
                    if (subgraph->num_of_outputs() > 0) {
                        no_subgraph_outputs = false;
                    }
                }
                // TODO: remove this check
                if (no_subgraph_outputs && node->get_schema().find("!") == std::string::npos) {
                    std::cout << "Node has no outputs: " << node->get_op_type() << " Skipping." << std::endl;
                    return;
                }
            }*/

            // Explore all inputs of node. Node may refer to input value that hasn't been created in the current scope.
            // But this value can be found in the outer scope, for this purpose we need to search node in
            // external_tensor_map as well

            auto raw_inputs = node->inputs();
            for (size_t i = 0; i < raw_inputs.size(); ++i) {
                auto input = node->input(i);
                if (tensor_map.find(input) == tensor_map.end()) {
                    // std::cout << "Level:" << LEVEL << " Trampoline for input index " << i << " with value " << input
                    // << "\n";
                    //   input refers value in the outer scope, need to create a new Parameter in the current scope
                    //   TODO: Connect outer scope and inner scope properly -- should be handled at the level of that
                    //   operation that introduced this nest of scopes (e.g. loop or if)
                    //   TODO: Eliminate duplication with the main code for Parameters creation
                    //   TODO: There is no real search for values in outer scope because we don't need to link the usage
                    //   and definition together at this point -- need to do that otherwise graph will fall apart
                    PartialShape ps = node->get_input_shape(i);
                    auto parameter = std::make_shared<opset8::Parameter>(node->get_input_type(i), ps);
                    // TODO: Missing get_input_transpose_order handling for not trivial layouts
                    tensor_map[input] = parameter;
                    // std::cout << "Parameter created\n";
                    // set name of parameter to the index of node in the model
                    parameter->get_output_tensor(0).add_names({std::to_string(input)});
                    parameters.push_back(parameter);
                    // std::cout << "External tensor: " << input << " node: " << external_tensor_map.at(input) <<
                    // std::endl;
                }
            }
            // std::cerr << "Node convert before translator: " << node->get_op_type() << ", schema: " <<
            // node->get_schema() << std::endl;

            auto context = NodeContext(node, &tensor_map, &parameters);
            auto converted_outputs = convert_node(&context);
            // std::cerr << "Node convert before outputs" << std::endl;

            auto mutated_t = context.get_mutated_tensors();
            mutated_tensors.insert(mutated_t.begin(), mutated_t.end());

            auto fw_outputs = node->outputs();
            // ops with subgraphs has more outputs
            FRONT_END_OP_CONVERSION_CHECK(fw_outputs.size() <= converted_outputs.size(),
                                          "Number of ",
                                          node->get_op_type(),
                                          " outputs greater then number of converted outputs.");

            // TODO: Make sure that mapping of fw_outputs to converted_outputs does always work
            // FIXME: Now it is not true for at least prim::Constant
            for (size_t i = 0; i < fw_outputs.size(); ++i) {
                size_t fw_tensor_id = node->output(i);
                if (tensor_map.find(fw_tensor_id) != tensor_map.end()) {
                    // std::cerr << "Duplicated producer for tensor with id = " << fw_tensor_id << " discovered at
                    // output "
                    //     << "port " << i << " of node " << node->kind().toQualString() << "\n";
                    throw std::runtime_error("Duplicated producer for PT value with unique ID: " +
                                             std::to_string(fw_tensor_id));
                }

                // Output shape of converted node should match the original output shape
                // std::cerr << "[ DEBUG ] PT output shape = " << get_ov_shape(fw_outputs[i]) << '\n';
                // std::cerr << "[ DEBUG ] OV output shape = " << converted_outputs[i].get_partial_shape() << '\n';
                // OV_FRONTEND_REQUIRE(get_ov_shape(fw_outputs[i]) == converted_outputs[i].get_partial_shape());

                tensor_map[fw_tensor_id] = converted_outputs[i];
                converted_outputs[i].get_tensor().add_names({std::to_string(fw_tensor_id)});
                // std::cout << "Level:" << LEVEL << " Added node: " << converted_outputs[i] << std::endl;
                //  std::cout << "Converted node output " << fw_tensor_id << ": " << converted_outputs[i] << std::endl;
            }
            // std::cout << "Node convert end" << std::endl;
        };

        OV_FRONTEND_REQUIRE(pytorch_model->get_subgraph_size() == 1);
        pytorch_model->visit_subgraph(0, node_visitor);
        // std::cout << "All nodes convert end" << std::endl;

        ResultVector results;
        // std::cerr << "Outputs:" << pytorch_model->num_of_outputs() << "\n";
        for (size_t i = 0; i < pytorch_model->num_of_outputs(); ++i) {
            size_t id = pytorch_model->output(i);
            // std::cerr << "Output:" << i << ": " << id << "\n";
            // std::cout << "value = " << id << '\n';
            // std::cout << "X\n";
            if (tensor_map.find(id) == tensor_map.end()) {
                // Not found in this scope, searching in the outer scope
                // TODO: do real search here, skipped for now

                auto parameter = std::make_shared<opset8::Parameter>(element::dynamic, PartialShape::dynamic());
                parameter->get_output_tensor(0).add_names({std::to_string(id)});
                parameters.push_back(parameter);
                tensor_map[id] = parameter;
                // std::cout << "Level:" << LEVEL << "Added new parameter based on external value " << id << "\n";
            }
            auto ov_output = tensor_map[id];
            // std::cout << "X\n";
            auto order = pytorch_model->get_output_transpose_order(i);
            // std::cout << "X\n";
            if (order.size() > 0 && !std::is_sorted(order.begin(), order.end())) {
                throw "Output strides have wrong order.";
            }
            // TODO: remove when all nodes has ids
            ov_output.add_names({std::to_string(id)});
            // std::cout << "X\n";
            // std::cout << ov_output << '\n';
            auto result = std::make_shared<opset8::Result>(ov_output);
            // std::cout << "X\n";
            results.push_back(result);
            // std::cerr << "Model result " << result << "\n";
        }

        // Since parameters can be added we need to list all current parameters
        std::set<size_t> param_names;
        for (auto param : parameters) {
            auto name = param->get_output_tensor(0).get_any_name();
            size_t input_idx = (size_t)std::stoll(name);
            param_names.insert(input_idx);
        }
        for (auto tensor : mutated_tensors) {
            if (param_names.count(tensor)) {
                OV_FRONTEND_REQUIRE(tensor_map.count(tensor));
                // model input was mutated we need to make a result for it
                results.push_back(std::make_shared<opset8::Result>(tensor_map.at(tensor)));
            }
        }
        // std::cout << "Y\n";

        /*for (size_t i = 0; i < parameters.size(); ++i) {
            auto parameter = parameters[i];
            // std::cerr << "parameter[" << i << "].shape = "
            //     << parameter->get_output_shape(0) << ", consumers: " <<
            //     parameter->output(0).get_target_inputs().size() << "\n";
        }*/
        // std::cout << "Convert end" << std::endl;
        // std::cout << "Number of values collected: " << tensor_map.size() << "\n";

        // std::cout << "=====Construct model start=====" << std::endl;
        /*std::cout << "=====Tensor map start=====" << std::endl;
        for (auto node : tensor_map) {
            std::cout << node.first << ": " << node.second.get_node_shared_ptr() << std::endl;
        }*/
        resulting_model = std::make_shared<ov::Model>(results, parameters);
        /*std::string m_name = "model_" + std::to_string(LEVEL) + "_" + std::to_string(NUMBER++);
        try {
            ov::serialize(resulting_model, m_name + ".xml", m_name + ".bin");
        } catch (...) {
            std::cout << "Exception happened during model serialization: " + m_name << std::endl;
        }*/
        // std::cout << "=====Construct model end=====" << std::endl;

        // Did a conversion in a nested scope to automatically remove any holders of nodes except those in the graph
    }

    // std::cout << "=====Convert model:" << LEVEL << " end=====" << std::endl;
    LEVEL--;
    return resulting_model;
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
