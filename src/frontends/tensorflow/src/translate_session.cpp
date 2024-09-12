// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "translate_session.hpp"

#include "helper_ops/enter.hpp"
#include "helper_ops/keep_in_graph_op.hpp"
#include "helper_ops/loop_cond.hpp"
#include "helper_ops/merge.hpp"
#include "helper_ops/next_iteration.hpp"
#include "helper_ops/switch.hpp"
#include "input_model.hpp"
#include "openvino/frontend/tensorflow/variable.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "tf_framework_node.hpp"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace ov::frontend::tensorflow;

namespace {
template <typename T>
std::vector<T> reorder_ops_by_names(const std::vector<std::string>& names, const std::vector<T>& ops) {
    if (names.empty()) {
        // in case unspecified names, return the initial order of operations
        return ops;
    }
    // some body graph input can turn to be a constant node
    FRONT_END_GENERAL_CHECK(names.size() >= ops.size(),
                            "[TensorFlow Frontend] Internal error: cannot perform reordering of operations. The number "
                            "of names mismatches the number of operations.");
    std::vector<T> resulted_ops(ops.size(), nullptr);

    for (const auto& op : ops) {
        const auto& op_name = op->get_friendly_name();
        auto iter = std::find(names.begin(), names.end(), op_name);
        FRONT_END_GENERAL_CHECK(iter != names.end(),
                                "[TensorFlow Frontend] Internal error: cannot perform reordering of operations. The "
                                "requested name is not found among operations.");
    }

    size_t ind = 0;
    for (const auto& name : names) {
        for (const auto& op : ops) {
            if (op->get_friendly_name() == name) {
                resulted_ops[ind] = op;
                ind++;
                break;
            }
        }
    }
    return resulted_ops;
};

/// \brief Adjusts names of the tensor by mapping internal names to user specific ones using the model signature
/// and mark unused tensor names that must be removed
/// \param[in] ov_output ov::Output<ov::Node> for which names set should be corrected
/// \param[in] saved_model_input_names Map of for input names
/// \param[in] saved_model_output_names Map of for output names
void adjust_saved_model_names(ov::Output<ov::Node>& ov_output,
                              const std::shared_ptr<std::map<std::string, std::string>>& saved_model_input_names,
                              const std::shared_ptr<std::map<std::string, std::string>>& saved_model_output_names) {
    // 1. check if it is the input or output tensor of the model
    // perform the adjustment only for the input and output tensors of the model
    auto param_node = ov::as_type_ptr<ov::op::v0::Parameter>(ov_output.get_node_shared_ptr());
    bool is_input_tensor = (param_node ? true : false);
    ov::ResultVector results;
    for (const auto& consumer : ov_output.get_target_inputs()) {
        if (const auto& result = ov::as_type_ptr<ov::op::v0::Result>(consumer.get_node()->shared_from_this())) {
            results.push_back(result);
        }
    }
    bool is_output_tensor = (results.size() > 0 ? true : false);
    if (!is_input_tensor && !is_output_tensor) {
        return;
    }

    // 2. find a set of clean-up names and aligned with the model signature
    const auto& tensor_names = ov_output.get_names();
    std::unordered_set<std::string> cleanup_names;
    bool signature_passed = true;
    if (is_input_tensor) {
        if (saved_model_input_names) {
            for (const auto& tensor_name : tensor_names) {
                if (saved_model_input_names->count(tensor_name) > 0) {
                    cleanup_names.insert(saved_model_input_names->at(tensor_name));
                    param_node->set_friendly_name(saved_model_input_names->at(tensor_name));
                }
            }
        } else {
            signature_passed = false;
        }
    }

    if (is_output_tensor) {
        if (saved_model_output_names) {
            std::vector<std::string> result_names;
            for (const auto& tensor_name : tensor_names) {
                if (saved_model_output_names->count(tensor_name) > 0) {
                    cleanup_names.insert(saved_model_output_names->at(tensor_name));
                    result_names.push_back(saved_model_output_names->at(tensor_name));
                }
            }
            // align the Result node names as many as possible
            // it is not bad if we remain it as is because OV API 2.0 relies only on tensor names
            size_t result_names_size = result_names.size();
            if (result_names_size > 0) {
                for (size_t ind = 0; ind < results.size(); ++ind) {
                    auto new_result_name = result_names[ind % result_names_size];
                    results[ind]->set_friendly_name(new_result_name);
                }
            }
        } else {
            signature_passed = false;
        }
    }

    // 3. set cleanup names to the tensor only if it is found in the signature
    // otherwise, the tensor corresponds to unused Parameter or Result nodes
    if (cleanup_names.size() > 0) {
        ov_output.set_names(cleanup_names);
    } else if (signature_passed) {
        // this is unused tensor that should be removed
        // because it not present in the signature
        ov_output.add_names({"saved_model_unused"});
    }
}

// it creates framework node and saves exception message in the node attribute
ov::OutputVector create_fw_node_with_exception(const std::shared_ptr<DecoderBase>& decoder,
                                               const ov::OutputVector& inputs,
                                               size_t num_outputs,
                                               const std::string& operation_name,
                                               const std::string& exception_message) {
    ov::op::util::FrameworkNodeAttrs attrs;
    attrs[FrameworkNode::failed_conversion_key] = exception_message;
    auto fw_node = std::make_shared<FrameworkNode>(decoder, inputs, num_outputs);
    fw_node->set_attrs(attrs);
    set_node_name(operation_name, fw_node);
    return fw_node->outputs();
}

size_t get_flat_index_by_name_and_id(const ov::frontend::NamedOutputVector& outputs,
                                     const std::string& name,
                                     size_t idx) {
    // Assume that if at least one output port has name, then all the ports should have names
    if (!outputs.empty() && !outputs.front().name.empty()) {
        // Producer has names in ports
        auto it = std::find_if(outputs.begin(), outputs.end(), [&](const ov::frontend::NamedOutput& x) {
            return name == x.name;
        });
        FRONT_END_GENERAL_CHECK(outputs.end() - it > ptrdiff_t(idx),
                                "There is no output port specified by name and index");
        FRONT_END_GENERAL_CHECK(it[idx].name == name,
                                "There is no output port with specified index in a group with specified name");
        return it - outputs.begin() + idx;
    } else {
        // There are no named ports in the producer node, so reference by name wouldn't work
        return idx;
    }
}

// create Parameter node that will produce given tensor
std::shared_ptr<ov::op::v0::Parameter> create_parameter_node_for_tensor(ov::Output<ov::Node> output_tensor) {
    auto param =
        std::make_shared<ov::op::v0::Parameter>(output_tensor.get_element_type(), output_tensor.get_partial_shape());
    param->output(0).set_names(output_tensor.get_names());
    output_tensor.replace(param->output(0));
    return param;
}

void fuse_loop_cond(std::shared_ptr<LoopCond>& loop_cond,
                    NameTensorMapPtr ov_tensors_map,
                    const std::vector<std::shared_ptr<Enter>>& enter_ops) {
    // ov_tensors_map maps a operation name to a vector of its output tensors
    auto node_name = loop_cond->get_friendly_name();
    // find key points for condition and body graphs
    FRONT_END_GENERAL_CHECK(loop_cond, "[TensorFlow Frontend] internal error: pointer to LoopCond node is nullptr");

    // extract condition and body graphs
    // scan LoopCond node vicinity
    // 1. LoopCond has just one output
    // walk through all consuming inputs that are expected to be only for Switch nodes
    std::vector<std::shared_ptr<Switch>> switch_nodes;
    for (const auto& consuming_input : loop_cond->get_output_target_inputs(0)) {
        auto switch_node = ov::as_type_ptr<Switch>(consuming_input.get_node()->shared_from_this());
        FRONT_END_GENERAL_CHECK(switch_node,
                                "[TensorFlow Frontend] internal error or inconsistent model: consumer of LoopCond "
                                "output is not Switch operation");
        switch_nodes.push_back(switch_node);
    }

    // collect all output tensors for Loop
    // the created Loop node outputs will be connected with ov_outputs
    size_t num_inputs = switch_nodes.size();
    FRONT_END_GENERAL_CHECK(num_inputs > 0,
                            "[TensorFlow Frontend] internal error: LoopCond node has no output Switch nodes");
    ov::OutputVector ov_outputs(num_inputs);
    // collect ov_inputs (a list of Tensors) that will provide input data for the created Loop node
    ov::OutputVector ov_inputs(num_inputs);
    ov::ParameterVector cond_params(num_inputs);
    ov::ParameterVector body_params(num_inputs);
    ov::OutputVector ov_body_outputs(num_inputs);
    std::vector<std::string> output_tensor_names(num_inputs);
    std::set<std::shared_ptr<Enter>> met_enter_ops;
    std::string frame_name;
    for (size_t ind = 0; ind < num_inputs; ++ind) {
        // Switch node has two outputs:
        // 0 (output_false) - interrupt the loop, 1 (output_true) - continue the loop
        // check if Exit node exists
        auto switch_node = switch_nodes[ind];
        FRONT_END_GENERAL_CHECK(
            switch_node->get_output_target_inputs(0).size() < 2,
            "[TensorFlow Frontend] internal error or inconsistent model: Switch node has more than one Exit nodes");
        if (switch_node->get_output_target_inputs(0).size() == 1) {
            auto exit_node = (*switch_node->get_output_target_inputs(0).begin()).get_node();
            ov_outputs[ind] = exit_node->output(0);
            output_tensor_names[ind] = exit_node->get_friendly_name() + ":0";
        }

        auto merge_node = ov::as_type_ptr<Merge>(switch_node->input_value(0).get_node_shared_ptr());
        FRONT_END_GENERAL_CHECK(merge_node,
                                "[TensorFlow Frontend] internal error or inconsistent model: Data for Switch node is "
                                "not produced by Merge node for While operation");

        // create Parameter node for condition graph
        cond_params[ind] = create_parameter_node_for_tensor(merge_node->output(0));
        body_params[ind] = create_parameter_node_for_tensor(switch_node->output(1));

        // check that Merge node has Enter and NextIteration producers
        auto enter = ov::as_type_ptr<Enter>(merge_node->input_value(0).get_node_shared_ptr());
        auto next_iteration = ov::as_type_ptr<NextIteration>(merge_node->input_value(0).get_node_shared_ptr());
        if (!enter) {
            enter = ov::as_type_ptr<Enter>(merge_node->input_value(1).get_node_shared_ptr());
        }
        if (!next_iteration) {
            next_iteration = ov::as_type_ptr<NextIteration>(merge_node->input_value(1).get_node_shared_ptr());
        }
        FRONT_END_GENERAL_CHECK(enter && next_iteration,
                                "[TensorFlow Frontend] internal error or inconsistent model: inputs of Merge node in "
                                "While sub-graph are not Enter and NextIteration");
        ov_inputs[ind] = enter->input_value(0);
        met_enter_ops.insert(enter);
        frame_name = enter->get_frame_name();

        // retrieve output tensor for body graph that is an input to NextIteration node
        std::string producer_name;
        size_t producer_output_port_idx;
        next_iteration->get_producer(producer_name, producer_output_port_idx);
        FRONT_END_GENERAL_CHECK(
            ov_tensors_map->count(producer_name) > 0,
            "[TensorFlow Frontend] internal error: NextIteration producer is not found in the tensor map");
        auto producer_outputs = ov_tensors_map->at(producer_name);
        FRONT_END_GENERAL_CHECK(
            producer_output_port_idx < producer_outputs.size(),
            "[TensorFlow Frontend] internal error: NextIteration producer has insufficient number of outputs");
        auto ov_body_output = producer_outputs[producer_output_port_idx].port;
        if (ov_body_output.get_node_shared_ptr() == switch_node) {
            // this is case when NextIteration node is connected with Switch node
            ov_body_outputs[ind] = body_params[ind]->output(0);
        } else {
            ov_body_outputs[ind] = ov_body_output;
        }
    }
    auto ov_cond_output = loop_cond->input_values();

    // insert additional inputs for future Loop node
    for (auto& enter : enter_ops) {
        if (met_enter_ops.find(enter) == met_enter_ops.end() && enter->get_frame_name() == frame_name) {
            ov_inputs.push_back(enter->input_value(0));
            auto additional_param = create_parameter_node_for_tensor(enter->output(0));
            cond_params.push_back(additional_param);
            body_params.push_back(additional_param);
        }
    }

    // create a copy of conditional graph
    auto cond_model = std::make_shared<ov::Model>(ov_cond_output, cond_params);
    auto body_model = std::make_shared<ov::Model>(ov_body_outputs, body_params);

    // check if condition model has NextIteration->Merge construction
    // if yes, it means we need to create separate condition for initial check prior to While execution
    // and separate one for Loop inside
    auto prior_cond_model = cond_model->clone();
    for (const auto& op : prior_cond_model->get_ordered_ops()) {
        auto merge = ov::as_type_ptr<Merge>(op);
        if (!merge) {
            continue;
        }

        auto next_iteration = ov::as_type_ptr<NextIteration>(merge->input_value(0).get_node_shared_ptr());
        if (!next_iteration) {
            next_iteration = ov::as_type_ptr<NextIteration>(merge->input_value(1).get_node_shared_ptr());
        }

        auto param_node = ov::as_type_ptr<ov::op::v0::Parameter>(merge->input_value(0).get_node_shared_ptr());
        if (!param_node) {
            param_node = ov::as_type_ptr<ov::op::v0::Parameter>(merge->input_value(1).get_node_shared_ptr());
        }

        if (!next_iteration || !param_node) {
            continue;
        }
        merge->output(0).replace(param_node->output(0));
    }

    // create condition model to inject inside Loop operaion
    auto cond_model_params = cond_model->get_parameters();
    for (const auto& op : cond_model->get_ordered_ops()) {
        auto merge = ov::as_type_ptr<Merge>(op);
        if (!merge) {
            continue;
        }

        auto next_iteration = ov::as_type_ptr<NextIteration>(merge->input_value(0).get_node_shared_ptr());
        if (!next_iteration) {
            next_iteration = ov::as_type_ptr<NextIteration>(merge->input_value(1).get_node_shared_ptr());
        }

        auto param_node = ov::as_type_ptr<ov::op::v0::Parameter>(merge->input_value(0).get_node_shared_ptr());
        if (!param_node) {
            param_node = ov::as_type_ptr<ov::op::v0::Parameter>(merge->input_value(1).get_node_shared_ptr());
        }

        if (!next_iteration || !param_node) {
            continue;
        }

        std::string producer_name;
        size_t producer_output_port_idx;
        next_iteration->get_producer(producer_name, producer_output_port_idx);
        FRONT_END_GENERAL_CHECK(
            ov_tensors_map->count(producer_name) > 0,
            "[TensorFlow Frontend] internal error: NextIteration producer is not found in the tensor map");
        auto producer_outputs = ov_tensors_map->at(producer_name);
        FRONT_END_GENERAL_CHECK(
            producer_output_port_idx < producer_outputs.size(),
            "[TensorFlow Frontend] internal error: NextIteration producer has insufficient number of outputs");
        auto next_iteration_output = producer_outputs[producer_output_port_idx].port;

        // create auxiliary body model having separate instances of ov::Nodes to avoid cycles in graph during Loop
        // construction node
        auto aux_cond_model =
            std::make_shared<ov::Model>(ov::OutputVector{next_iteration_output}, body_params)->clone();
        auto aux_cond_params = aux_cond_model->get_parameters();
        auto aux_cond_results = aux_cond_model->get_results();
        auto params_size = aux_cond_params.size();
        // insert the auxiliary body model into condition model
        for (size_t param_ind = 0; param_ind < params_size; ++param_ind) {
            auto cond_param = cond_model_params[param_ind];
            aux_cond_params[param_ind]->output(0).replace(cond_model_params[param_ind]->output(0));
        }
        merge->output(0).replace(aux_cond_results[0]->input_value(0));
    }

    auto loop_node = create_loop_for_tf_while(node_name, body_model, cond_model, ov_inputs, prior_cond_model);
    auto loop_model = std::make_shared<ov::Model>(loop_node->outputs());

    size_t loop_node_output_size = loop_node->get_output_size();
    FRONT_END_GENERAL_CHECK(loop_node_output_size == num_inputs,
                            "[TensorFlow Frontend] internal error: the created Loop node to replace TF1 While has "
                            "unexpected number of outputs");
    for (size_t output_ind = 0; output_ind < loop_node_output_size; ++output_ind) {
        auto producer_node = ov_outputs[output_ind].get_node_shared_ptr();
        if (producer_node) {
            std::string producer_name = producer_node->get_friendly_name();
            size_t producer_output_port_idx = ov_outputs[output_ind].get_index();
            // work only for non-empty ov::Output<ov::Node>
            ov_outputs[output_ind].replace(loop_node->output(output_ind));
            ov_outputs[output_ind].set_names({output_tensor_names[output_ind]});
            if (ov_tensors_map->count(producer_name) &&
                producer_output_port_idx < ov_tensors_map->at(producer_name).size()) {
                ov_tensors_map->at(producer_name)[producer_output_port_idx] = ov_outputs[output_ind];
            }
        }
    }
}
}  // namespace

TranslateSession::TranslateSession(const ov::frontend::InputModel::Ptr& input_model,
                                   const std::shared_ptr<TranslatorDictionaryType>& translator_map,
                                   const std::string& model_name)
    : m_input_model(input_model),
      m_translator_map(translator_map),
      m_model_name(model_name),
      m_ov_model(nullptr),
      m_cached_body_models(std::make_shared<CachedBodyModelsType>()),
      m_variables_map(std::make_shared<VariableMap>()) {}

std::shared_ptr<ov::Model> TranslateSession::get_converted_model() {
    if (m_ov_model) {
        return m_ov_model;
    }
    translate_graph(m_input_model, m_ov_model);
    return m_ov_model;
}

void TranslateSession::translate_graph(const ov::frontend::InputModel::Ptr& input_model,
                                       std::shared_ptr<ov::Model>& ov_model) {
    NameTensorMapPtr ov_tensors_map = std::make_shared<NameTensorMap>();
    VariableMap::Ptr ov_variables_map = get_variable_map();
    ControlDepsMap control_deps_map;
    std::vector<std::shared_ptr<LoopCond>> loop_cond_ops;
    std::vector<std::shared_ptr<Enter>> enter_ops;

    ov::ParameterVector params;
    ov::ResultVector results;
    ov::SinkVector sinks;
    const auto& model_tf = std::dynamic_pointer_cast<InputModel>(input_model);
    FRONT_END_GENERAL_CHECK(model_tf, "nullptr for InputModel is given for translation into OV Model");
    const auto& operation_places = model_tf->get_op_places();
    const auto& model_inputs = model_tf->get_inputs();
    const auto& model_outputs = model_tf->get_outputs();
    const auto& model_frozen_inputs = model_tf->get_tensor_values();
    const auto& saved_model_inputs = model_tf->get_saved_model_input_names();
    const auto& saved_model_outputs = model_tf->get_saved_model_output_names();
    bool is_body_graph = (model_tf->get_input_names().size() > 0);

    // fill ng_op_map with Constant outputs for frozen inputs
    for (const auto& frozen_input : model_frozen_inputs) {
        const auto& frozen_input_name = frozen_input.first;
        const auto& frozen_input_value = frozen_input.second;
        FRONT_END_GENERAL_CHECK(ov_tensors_map->count(frozen_input_name) == 0,
                                "Input with frozen value has been already met: " + frozen_input_name);
        (*ov_tensors_map)[frozen_input_name] = {frozen_input_value};
    }
    // create parameter nodes for all tensor places corresponding to inputs
    for (const auto& input_place : model_inputs) {
        FRONT_END_GENERAL_CHECK(input_place->get_names().size() == 1, "Input place must have one name.");
        auto input_name = input_place->get_names()[0];
        if (ov_tensors_map->count(input_name)) {
            // probably this input is frozen
            continue;
        }
        const auto& input_tensor_place = std::dynamic_pointer_cast<TensorPlace>(input_place);
        auto input_shape = input_tensor_place->get_partial_shape();
        auto input_type = input_tensor_place->get_element_type();
        auto input_op_name = input_tensor_place->get_operation_name();

        // in case of cutting graph, types of custom inputs can be dynamic,
        // according to MO help, fp32 is used by default in such cases
        if (input_type == element::dynamic) {
            input_type = element::f32;
        }

        if (const auto& input_var = model_tf->get_variable(input_place)) {
            (*ov_tensors_map)[input_name] = {NamedOutput(input_var->output(0))};
        } else {
            auto param = std::make_shared<ov::op::v0::Parameter>(input_type, input_shape);
            param->set_friendly_name(input_name);
            set_out_name(input_name, param);
            params.push_back(param);
            (*ov_tensors_map)[input_op_name] = {NamedOutput(param)};
        }
    }

    // create the OV ops from TensorFlow ops
    std::vector<std::string> data_producer_names;
    for (const auto& operation_place : operation_places) {
        auto operation_decoder = operation_place->get_decoder();
        auto operation_name = operation_place->get_names()[0];
        // output for parameter nodes has been already generated
        if (ov_tensors_map->count(operation_name)) {
            continue;
        }

        // prepare a list of OV node inputs for each node
        ov::OutputVector ov_inputs;
        size_t operation_input_size = operation_decoder->get_input_size();
        std::vector<std::string> control_dependencies_names;

        if (operation_decoder->get_op_type() == "NextIteration") {
            // we expect no inputs for NextIteration because we break-up the cycle in InputModel
            operation_input_size = 0;
        }
        for (size_t input_port_idx = 0; input_port_idx < operation_input_size; ++input_port_idx) {
            std::string producer_name;
            size_t producer_port_idx;
            try {
                std::string producer_port_name;
                operation_decoder->get_input_node(input_port_idx, producer_name, producer_port_name, producer_port_idx);
                if (!producer_port_name.empty()) {
                    producer_port_idx = get_flat_index_by_name_and_id(ov_tensors_map->at(producer_name),
                                                                      producer_port_name,
                                                                      producer_port_idx);
                }
            } catch (const std::exception&) {
                FRONT_END_THROW("[ ERROR ] Exception happened when preparing input " + std::to_string(input_port_idx) +
                                " for op '" + operation_decoder->get_op_name() + "', expected input name: '" +
                                producer_name + "', expected input port index: " + std::to_string(producer_port_idx) +
                                '\n');
            }

            // skip conditional edges that must be resolved before operation translation
            // now we can meet them because we still work with TensorFlow protobuf
            if (is_conditional_edge(producer_name)) {
                // control dependency contains "^" in the beginning
                control_dependencies_names.push_back(producer_name.substr(1));
                continue;
            } else {
                // save all node names producing data
                data_producer_names.push_back(producer_name);
            }

            // TODO: re-implement the logic below once Place graph structure is implemented
            // Using Place graph structure (OpPlace, In/OutPortPlace places and their connections) can give
            // names of ports and operations that can be used for further check about existence in ng_op_map

            // check if output vector for places have been already defined and the order of this check is important
            // it moves from places corresponding to input port of the current operation node to output port of original
            // producers
            if (ov_tensors_map->count(std::to_string(input_port_idx) + ":" + operation_name)) {
                const auto& input_outputs_vector =
                    ov_tensors_map->at(std::to_string(input_port_idx) + ":" + operation_name);
                FRONT_END_GENERAL_CHECK(input_outputs_vector.size() == 1,
                                        "Input created with pruning must have one output");
                ov_inputs.push_back(input_outputs_vector.at(0).port);
            } else if (ov_tensors_map->count(producer_name + ":" + std::to_string(producer_port_idx))) {
                const auto& input_outputs_vector =
                    ov_tensors_map->at(producer_name + ":" + std::to_string(producer_port_idx));
                FRONT_END_GENERAL_CHECK(input_outputs_vector.size() == 1,
                                        "Input created with pruning must have one output");
                ov_inputs.push_back(input_outputs_vector.at(0).port);
            } else if (ov_tensors_map->count(producer_name)) {
                const auto& input_outputs_vector = ov_tensors_map->at(producer_name);
                if (input_outputs_vector.size() <= producer_port_idx) {
                    auto producer_node = input_outputs_vector[0].port.get_node_shared_ptr();
                    if (std::dynamic_pointer_cast<FrameworkNode>(producer_node)) {
                        // FrameworkNode node does not know in advance how many output ports will be used
                        // so we can increase number of outputs by demand
                        producer_node->set_output_type(producer_port_idx, element::dynamic, PartialShape::dynamic());
                        // update output vector in node map
                        (*ov_tensors_map)[producer_name] = named_from_indexed(producer_node->outputs());
                    }
                }
                FRONT_END_GENERAL_CHECK(input_outputs_vector.size() > producer_port_idx,
                                        "Input created with pruning must have one output");
                ov_inputs.push_back(input_outputs_vector.at(producer_port_idx).port);
            } else {
                FRONT_END_GENERAL_CHECK(false,
                                        "No input is found for node \"" + operation_name + "\" by port " +
                                            std::to_string(producer_port_idx));
            }
        }

        // update variables state map for the current node
        ov_variables_map->initialize_variable_state_map_for_node(control_dependencies_names,
                                                                 data_producer_names,
                                                                 operation_name);

        // generate OV node output vector for the current operation node
        NamedOutputVector ov_outputs;
        auto operation_type = operation_decoder->get_op_type();
        if (m_translator_map->count(operation_type)) {
            try {
                auto translator = m_translator_map->at(operation_decoder->get_op_type());
                NodeContext node_context(operation_decoder, ov_inputs, ov_variables_map, this);
                ov_outputs = translator(node_context);
            } catch (const std::exception& ex) {
                // save the root-cause of the translation failure
                const auto fw_outs = create_fw_node_with_exception(operation_decoder,
                                                                   ov_inputs,
                                                                   operation_place->get_output_ports().size(),
                                                                   operation_name,
                                                                   ex.what());
                ov_outputs = named_from_indexed(fw_outs);
            } catch (...) {
                // save unknown exception type
                const auto fw_outs = create_fw_node_with_exception(operation_decoder,
                                                                   ov_inputs,
                                                                   operation_place->get_output_ports().size(),
                                                                   operation_name,
                                                                   "Unknown exception type");
                ov_outputs = named_from_indexed(fw_outs);
            }

            for (auto output : ov_outputs) {
                auto node = output.port.get_node_shared_ptr();
                // We can't add all Sink operations to sinks vector, as there can be a FrameworkNode,
                // which we might need to remove from graph
                if (ov::as_type_ptr<KeepInGraphOp>(node)) {
                    sinks.push_back(std::dynamic_pointer_cast<ov::op::Sink>(node));
                } else {
                    auto multi_subgraph = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp>(node);
                    if (multi_subgraph) {
                        for (const auto& body_model : multi_subgraph->get_functions()) {
                            if (body_model->get_sinks().size()) {
                                sinks.push_back(std::dynamic_pointer_cast<ov::op::Sink>(multi_subgraph));
                                break;
                            }
                        }
                    }
                }
            }
        } else if (auto body_ov_model = get_body_ov_model(operation_type, ov_inputs)) {
            OutputVector indexed_ov_outputs;
            inject_body_model(body_ov_model, operation_type, ov_inputs, indexed_ov_outputs);

            // set output tensor names
            for (size_t idx = 0; idx < indexed_ov_outputs.size(); ++idx) {
                indexed_ov_outputs[idx].get_tensor().set_names({operation_name + ":" + std::to_string(idx)});
            }
            ov_outputs = named_from_indexed(indexed_ov_outputs);
        } else {
            // continue translation by replacing with FrameworkNode
            // for example, it helps auto-pruning to be triggered on later nodes
            auto fw_node = std::make_shared<FrameworkNode>(operation_decoder,
                                                           ov_inputs,
                                                           operation_place->get_output_ports().size());
            set_node_name(operation_name, fw_node);
            ov_outputs = named_from_indexed(fw_node->outputs());
        }

        // save LoopCond operations in topological order for further fusing
        if (ov_outputs.size() == 1 && as_type_ptr<LoopCond>(ov_outputs[0].port.get_node_shared_ptr())) {
            loop_cond_ops.push_back(as_type_ptr<LoopCond>(ov_outputs[0].port.get_node_shared_ptr()));
        } else if (ov_outputs.size() == 1 && as_type_ptr<Enter>(ov_outputs[0].port.get_node_shared_ptr())) {
            enter_ops.push_back(as_type_ptr<Enter>(ov_outputs[0].port.get_node_shared_ptr()));
        } else if (ov_outputs.size() == 1 && as_type_ptr<NextIteration>(ov_outputs[0].port.get_node_shared_ptr())) {
            std::string producer_name;
            size_t producer_output_port_idx;
            operation_place->get_next_iteration_back_edge(producer_name, producer_output_port_idx);
            auto next_iteration = as_type_ptr<NextIteration>(ov_outputs[0].port.get_node_shared_ptr());
            next_iteration->set_producer(producer_name, producer_output_port_idx);
        }

        // create input control dependencies set for the current operation node
        std::set<ov::Output<ov::Node>> input_control_deps;
        for (const auto& control_dep_name : control_dependencies_names) {
            if (control_deps_map.count(control_dep_name) > 0) {
                const auto& input_control_dep = control_deps_map[control_dep_name];
                input_control_deps.insert(input_control_dep.cbegin(), input_control_dep.cend());
            }
        }
        // register output control dependencies in control dependencies map
        std::set<ov::Output<ov::Node>> output_control_deps;
        if (propagate_conditional_flow(ov_inputs, ov_outputs, input_control_deps, output_control_deps)) {
            control_deps_map[operation_name] = output_control_deps;
        }

        // register OV node outputs in the map for new operation node
        for (const auto& output : ov_outputs) {
            if (auto result = as_type_ptr<ov::op::v0::Result>(output.port.get_node_shared_ptr())) {
                // do not add RetVal type operation to ng_op_map
                results.push_back(result);
            } else {
                auto param = as_type_ptr<ov::op::v0::Parameter>(output.port.get_node_shared_ptr());
                // avoid duplicating Parameter nodes if they are already in the Parameters vector
                if (param && std::find(params.begin(), params.end(), param) == params.end() && !is_body_graph) {
                    params.push_back(param);
                }
                (*ov_tensors_map)[operation_name].push_back(output);
            }
        }
    }

    // create Result nodes for all model outputs
    if (results.empty()) {
        for (const auto& model_output : model_outputs) {
            auto model_output_tensor_place = std::dynamic_pointer_cast<TensorPlace>(model_output);
            auto model_output_name = model_output_tensor_place->get_names()[0];
            std::string operation_name;
            std::string port_type;
            size_t port_index;
            ov::frontend::tensorflow::extract_operation_name_and_port(model_output_name,
                                                                      operation_name,
                                                                      port_index,
                                                                      port_type);

            if (port_type == "none") {
                for (const auto& node_output : indexed_from_named((*ov_tensors_map)[operation_name])) {
                    auto result_node = std::make_shared<ov::op::v0::Result>(node_output);
                    // to be aligned with Legacy Frontend we set a name along with output port index
                    // though, the Result name is not used in the OV API 2.0 but it is checked in MO args tests
                    result_node->set_friendly_name(model_output_name + ":0");
                    results.push_back(result_node);
                }
            } else if (port_type == "out") {
                const auto& node_outputs = indexed_from_named((*ov_tensors_map)[operation_name]);
                if (node_outputs.size() > port_index) {
                    auto result_node = std::make_shared<ov::op::v0::Result>(node_outputs[port_index]);
                    result_node->set_friendly_name(model_output_name);
                    results.push_back(result_node);
                }
            } else if (port_type == "in") {
                // TODO: avoid this traversing by having a map for OpPlace objects, for example
                std::shared_ptr<OpPlace> operation_place = nullptr;
                for (const auto& op_place : operation_places) {
                    FRONT_END_GENERAL_CHECK(!op_place->get_names().empty(), "No names for OpPlace found.");
                    if (op_place->get_names()[0] == operation_name) {
                        operation_place = op_place;
                    }
                }
                FRONT_END_GENERAL_CHECK(operation_place, "There is no operation place with a name: " + operation_name);
                auto operation_decoder = operation_place->get_decoder();

                // get to know a producer node and by which its output port data is generated
                std::string producer_name;
                std::string producer_port_name;
                size_t producer_port_idx;
                try {
                    operation_decoder->get_input_node(port_index, producer_name, producer_port_name, producer_port_idx);
                    if (!producer_port_name.empty()) {
                        producer_port_idx = get_flat_index_by_name_and_id((*ov_tensors_map)[producer_name],
                                                                          producer_port_name,
                                                                          producer_port_idx);
                    }
                } catch (const std::exception&) {
                    FRONT_END_THROW("[ ERROR ] Exception happened when preparing input " + std::to_string(port_index) +
                                    " for op '" + operation_decoder->get_op_name() + "', expected input name: '" +
                                    producer_name +
                                    "', expected input port index: " + std::to_string(producer_port_idx) + '\n');
                }

                // add Result node for this producer output port
                const auto& node_outputs = indexed_from_named((*ov_tensors_map)[producer_name]);
                FRONT_END_GENERAL_CHECK(node_outputs.size() > producer_port_idx,
                                        "Output port with index " + std::to_string(producer_port_idx) + " of " +
                                            producer_name + "node specified as custom output does not exist");
                auto result_node = std::make_shared<ov::op::v0::Result>(node_outputs[producer_port_idx]);
                // to be aligned with Legacy Frontend we set a name of the output tensor name
                // of the producer to the Result node
                // though, the Result name is not used in the OV API 2.0 but it is checked in MO args tests
                result_node->set_friendly_name(producer_name + ":" + std::to_string(producer_port_idx));
                results.push_back(result_node);
            }
        }
    }

    // TODO: it may be redundant step since models_output is filled in InputModel constructor
    // find all terminal nodes in OV graph to complete list of results
    if (results.empty()) {
        for (const auto& node_output_vector : *ov_tensors_map) {
            for (size_t output_ind = 0; output_ind < node_output_vector.second.size(); ++output_ind) {
                auto output = node_output_vector.second[output_ind].port;
                if (output.get_target_inputs().empty() &&
                    !std::dynamic_pointer_cast<ov::op::v0::Result>(output.get_node_shared_ptr())) {
                    auto model_output_name =
                        output.get_node_shared_ptr()->get_friendly_name() + ":" + std::to_string(output_ind);
                    auto result_node = std::make_shared<ov::op::v0::Result>(output);
                    result_node->set_friendly_name(model_output_name);
                    results.push_back(result_node);
                }
            }
        }
    }

    if (saved_model_inputs || saved_model_outputs) {
        // only SavedModel and MetaGraph models have mapping from internal tensor names to user specific ones
        // for example, serving_default_input_name:0 maps to input_name
        // we need to re-write input and output internal tensor names to user specific

        // it makes sense to use set because Parameter and Results nodes may have the common tensor
        std::set<ov::Output<ov::Node>> ov_tensors;
        for (const auto& param : params) {
            ov_tensors.insert(param->output(0));
        }
        for (const auto& result : results) {
            ov_tensors.insert(result->input_value(0));
        }

        // it iterates through these tensors and adjusts their names
        // by remaining only user specific names or mark as unused tensor (produced by TensorFlow)
        for (auto ov_tensor : ov_tensors) {
            adjust_saved_model_names(ov_tensor, saved_model_inputs, saved_model_outputs);
        }
    }

    // reorder Parameter and Result nodes according to the requested order
    // of input and output names from the original model
    // during translation and topologically sorting this order could be lost
    auto input_names = model_tf->get_input_names();
    auto output_names = model_tf->get_output_names();
    ov::ParameterVector ordered_params = reorder_ops_by_names(input_names, params);
    ov::ResultVector ordered_results = reorder_ops_by_names(output_names, results);

    // before adding Result nodes to terminal nodes
    // it fuses TF1 Control flow based While operation to Loop operation
    // it needs to perform this in the reverse order
    std::reverse(loop_cond_ops.begin(), loop_cond_ops.end());
    for (auto& loop_cond_op : loop_cond_ops) {
        fuse_loop_cond(loop_cond_op, ov_tensors_map, enter_ops);
    }
    ov_model = std::make_shared<ov::Model>(ordered_results, sinks, ordered_params, m_model_name);
}

std::shared_ptr<ov::Model> TranslateSession::get_body_ov_model(const std::string& body_graph_name,
                                                               const ov::OutputVector& ov_inputs,
                                                               bool clear_names) {
    std::shared_ptr<ov::Model> body_model = nullptr;
    auto input_model = std::dynamic_pointer_cast<InputModel>(m_input_model);
    std::vector<ov::PartialShape> input_shapes;
    input_shapes.reserve(ov_inputs.size());
    std::vector<ov::element::Type> input_types;
    input_types.reserve(ov_inputs.size());
    for (const auto& ov_input : ov_inputs) {
        input_shapes.push_back(ov_input.get_partial_shape());
        input_types.push_back(ov_input.get_element_type());
    }
    CachedBodyModelSignature body_model_signature{body_graph_name, input_shapes, input_types};

    if (m_cached_body_models->count(body_model_signature)) {
        // check if such body graph has been converted before
        // re-use it from the cache for further injection

        // create new instance of the required body model
        // since it will be modified by injection
        auto cached_body_model = m_cached_body_models->at(body_model_signature);
        body_model = cached_body_model->clone();
    } else if (auto body_input_model = input_model->get_body_input_model(body_graph_name)) {
        // set input shapes and types for InputModel of the body graph
        // it allows to get more optimized model after the conversion,
        // for example, to get less sub-graphs with ShapeOf and Convert operations
        // input names set an order of body graph inputs
        auto input_names = body_input_model->get_input_names();
        auto body_inputs = body_input_model->get_inputs();
        size_t int_num_inputs = body_inputs.size();
        size_t ext_num_inputs = ov_inputs.size();
        FRONT_END_GENERAL_CHECK(int_num_inputs <= ext_num_inputs,
                                "[TensorFlow Frontend] internal error: a number of external and "
                                "internal inputs for a body graph mismatch");
        FRONT_END_GENERAL_CHECK(input_names.size() == ext_num_inputs,
                                "[TensorFlow Frontend] internal error: a number of body graph names and external "
                                "inputs to body must match");
        for (size_t input_ind = 0; input_ind < ext_num_inputs; ++input_ind) {
            auto required_input_name = input_names[input_ind];
            bool is_found_body_input = false;
            size_t body_found_ind = 0;
            for (size_t internal_ind = 0; internal_ind < int_num_inputs; ++internal_ind) {
                auto body_input_place = body_inputs[internal_ind];
                auto body_input_names = body_input_place->get_names();
                if (std::find(body_input_names.begin(), body_input_names.end(), required_input_name) !=
                    body_input_names.end()) {
                    is_found_body_input = true;
                    body_found_ind = internal_ind;
                    break;
                }
            }
            if (is_found_body_input) {
                auto body_input_place = body_inputs[body_found_ind];
                // if body input with required name is found, set its type
                if (input_types[input_ind].is_static()) {
                    body_input_model->set_element_type(body_input_place, input_types[input_ind]);
                }
                if (input_shapes[input_ind].rank().is_static()) {
                    body_input_model->set_partial_shape(body_input_place, input_shapes[input_ind]);
                }
                // set variable to the corresponding place
                // it is needed to propogate variable values inside a body graph like HashTable
                // without this, conversion of LookupTableFind operations will not be possible
                if (const auto& input_var = as_type_ptr<Variable>(ov_inputs[input_ind].get_node_shared_ptr())) {
                    body_input_model->set_variable(body_input_place, input_var);
                }
            }
        }

        // try to find a function by name in the model library
        translate_graph(body_input_model, body_model);
        // save new instance of body_model in the cache of body models
        // before its injection into the parent graph

        // before caching, erase tensor names from the body graph
        // otherwise, it can lead tensor names conflicts
        if (clear_names) {
            for (const auto& op : body_model->get_ordered_ops()) {
                for (size_t ind = 0; ind < op->get_output_size(); ++ind) {
                    op->get_output_tensor(ind).set_names({});
                }
            }
        }

        auto cached_body_model = body_model->clone();
        update_cached_body_models(body_model_signature, cached_body_model);
    }
    return body_model;
}
