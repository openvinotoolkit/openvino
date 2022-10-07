// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/tensorflow/frontend.hpp"

#include "graph_iterator_proto.hpp"
#include "helper_transforms/block_lstm_replacer.hpp"
#include "helper_transforms/embedding_segments_feature_fusing.hpp"
#include "helper_transforms/gru_block_cell_replacer.hpp"
#include "input_model.hpp"
#include "op_table.hpp"
#include "openvino/frontend/tensorflow/extension/conversion.hpp"
#include "openvino/frontend/tensorflow/graph_iterator.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"
#include "pass/transpose_sinking.hpp"
#include "so_extension.hpp"
#include "tf_framework_node.hpp"
#include "utils.hpp"

using namespace ov::frontend::tensorflow;

namespace {
void translate_framework_node(const std::shared_ptr<FrameworkNode>& node,
                              const TranslatorDictionaryType& op_translators) {
    auto type = node->get_op_type();

    const auto& TRANSLATE_OP_MAP = op_translators;
    auto translator_it = TRANSLATE_OP_MAP.find(type);
    FRONT_END_OP_CONVERSION_CHECK(translator_it != TRANSLATE_OP_MAP.end(), "No translator found for ", type, " node.");

    ov::OutputVector ng_inputs = node->input_values();
    NodeContext node_ctx(node->get_decoder(), ng_inputs);
    auto new_node_outputs = translator_it->second(node_ctx);

    auto new_output = new_node_outputs.begin();
    auto old_outputs = node->outputs();
    auto old_output = old_outputs.begin();

    for (; new_output != new_node_outputs.end() && old_output != old_outputs.end(); ++old_output, ++new_output) {
        old_output->replace(*new_output);
    }
}
}  // namespace

FrontEnd::FrontEnd() : m_op_translators(tensorflow::op::get_supported_ops()) {}

void FrontEnd::translate_graph(const ov::frontend::InputModel::Ptr& model,
                               const std::string& model_name,
                               bool fail_fast,
                               bool no_conversion,
                               std::shared_ptr<ov::Model>& ng_function) const {
    // a map from operation names to generated OV Output<TFNodeDecoder>
    tensorflow::OpMap ng_op_map;

    ov::ParameterVector params;
    ov::ResultVector results;
    const auto& model_tf = std::dynamic_pointer_cast<InputModel>(model);
    FRONT_END_GENERAL_CHECK(model_tf, "nullptr for InputModel is given for translation into OV Model");
    const auto& operation_places = model_tf->get_op_places();
    const auto& model_inputs = model_tf->get_inputs();
    const auto& model_outputs = model_tf->get_outputs();
    const auto& model_frozen_inputs = model_tf->get_tensor_values();
    std::map<const std::string, const std::function<ov::OutputVector(const NodeContext&)>> translate_map;

    const auto& TRANSLATE_OP_MAP = m_op_translators;
    if (no_conversion) {
        const std::set<std::string> required_types{"Placeholder", "NoOp"};
        for (const auto& name : required_types) {
            translate_map.emplace(name, TRANSLATE_OP_MAP.at(name));
        }
    } else {
        translate_map.insert(TRANSLATE_OP_MAP.begin(), TRANSLATE_OP_MAP.end());
    }

    // fill ng_op_map with Constant outputs for frozen inputs
    for (const auto& frozen_input : model_frozen_inputs) {
        const auto& frozen_input_name = frozen_input.first;
        const auto& frozen_input_value = frozen_input.second;
        FRONT_END_GENERAL_CHECK(ng_op_map.count(frozen_input_name) == 0,
                                "Input with frozen value has been already met: " + frozen_input_name);
        ng_op_map[frozen_input_name] = {frozen_input_value};
    }
    // create parameter nodes for all tensor places corresponding to inputs
    for (const auto& input_place : model_inputs) {
        FRONT_END_GENERAL_CHECK(input_place->get_names().size() == 1, "Input place must have one name.");
        auto input_name = input_place->get_names()[0];
        if (ng_op_map.count(input_name)) {
            // probably this input is frozen
            continue;
        }
        const auto& input_tensor_place = std::dynamic_pointer_cast<TensorPlace>(input_place);
        auto input_shape = input_tensor_place->get_partial_shape();
        auto input_type = input_tensor_place->get_element_type();

        auto param = std::make_shared<ov::opset8::Parameter>(input_type, input_shape);
        set_node_name(input_name, param);
        params.push_back(param);
        ng_op_map[input_name] = {param};
    }

    // create the OV ops from TensorFlow ops
    for (const auto& operation_place : operation_places) {
        auto operation_decoder = operation_place->get_decoder();
        auto operation_name = operation_place->get_names()[0];
        // output for parameter nodes has been already generated
        if (ng_op_map.count(operation_name)) {
            continue;
        }

        // prepare a list of OV node inputs for each node
        ov::OutputVector ng_inputs;
        for (size_t input_port_idx = 0; input_port_idx < operation_decoder->get_input_size(); ++input_port_idx) {
            // TODO: Implement more general approach. Skipping Constants that have input edges
            if (operation_decoder->get_op_type() == "Const") {
                break;
            }
            std::string producer_name;
            size_t producer_port_idx;
            try {
                operation_decoder->get_input_node(input_port_idx, producer_name, producer_port_idx);
            } catch (const std::exception&) {
                FRONT_END_THROW("[ ERROR ] Exception happened when preparing input " + std::to_string(input_port_idx) +
                                " for op '" + operation_decoder->get_op_name() + "', expected input name: '" +
                                producer_name + "', expected input port index: " + std::to_string(producer_port_idx) +
                                '\n');
            }

            // skip conditional edges that must be resolved before operation translation
            // now we can meet them because we still work with TensorFlow protobuf
            if (is_conditional_edge(producer_name)) {
                continue;
            }

            // TODO: re-implement the logic below once Place graph structure is implemented
            // Using Place graph structure (OpPlace, In/OutPortPlace places and their connections) can give
            // names of ports and operations that can be used for further check about existence in ng_op_map

            // check if output vector for places have been already defined and the order of this check is important
            // it moves from places corresponding to input port of the current operation node to output port of original
            // producers
            if (ng_op_map.count(std::to_string(input_port_idx) + ":" + operation_name)) {
                const auto& input_outputs_vector = ng_op_map.at(std::to_string(input_port_idx) + ":" + operation_name);
                FRONT_END_GENERAL_CHECK(input_outputs_vector.size() == 1,
                                        "Input created with pruning must have one output");
                ng_inputs.push_back(input_outputs_vector.at(0));
            } else if (ng_op_map.count(producer_name + ":" + std::to_string(producer_port_idx))) {
                const auto& input_outputs_vector =
                    ng_op_map.at(producer_name + ":" + std::to_string(producer_port_idx));
                FRONT_END_GENERAL_CHECK(input_outputs_vector.size() == 1,
                                        "Input created with pruning must have one output");
                ng_inputs.push_back(input_outputs_vector.at(0));
            } else if (ng_op_map.count(producer_name)) {
                const auto& input_outputs_vector = ng_op_map.at(producer_name);
                FRONT_END_GENERAL_CHECK(input_outputs_vector.size() > producer_port_idx,
                                        "Input created with pruning must have one output");
                ng_inputs.push_back(input_outputs_vector.at(producer_port_idx));
            } else {
                FRONT_END_GENERAL_CHECK(false,
                                        "No input is found for node \"" + operation_name + "\" by port " +
                                            std::to_string(producer_port_idx));
            }
        }

        // generate OV node output vector for the current operation node
        ov::OutputVector ng_outputs;
        try {
            FRONT_END_OP_CONVERSION_CHECK(translate_map.count(operation_decoder->get_op_type()),
                                          "No translator found for " + operation_decoder->get_op_type() + " node.");
            auto op_fun = &(translate_map[operation_decoder->get_op_type()]);
            // NodeContext node_context(ng_inputs, operation_decoder, model_inputs);
            // TODO: Check why NodeContextNew doesn't have ngOutputVector ng_inputs input in constructor
            NodeContext node_context(operation_decoder, ng_inputs);
            // generate OV node output vector using translator for given operation type
            ng_outputs = (*op_fun)(node_context);
        } catch (...) {
            if (fail_fast) {
                // re-throw any exception
                throw;
            } else {
                auto ng_node = std::make_shared<FrameworkNode>(operation_decoder,
                                                               ng_inputs,
                                                               operation_place->get_output_ports().size());
                set_node_name(operation_name, ng_node);
                ng_outputs = ng_node->outputs();
            }
        }

        // register OV node outputs in the map for new operation node
        for (const auto& output : ng_outputs) {
            if (auto result = std::dynamic_pointer_cast<ov::opset8::Result>(output.get_node_shared_ptr())) {
                // do not add RetVal type operation to ng_op_map
                results.push_back(result);
            } else {
                auto param = std::dynamic_pointer_cast<ov::opset8::Parameter>(output.get_node_shared_ptr());
                // avoid duplicating Parameter nodes if they are already in the Parameters vector
                if (param && operation_decoder->get_op_type() != "Identity" &&
                    std::find(params.begin(), params.end(), param) == params.end()) {
                    params.push_back(param);
                }
                ng_op_map[operation_name].push_back(output);
            }
        }
    }

    // create Result nodes for all model outputs
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
            for (const auto& node_output : ng_op_map[operation_name]) {
                auto result_node = std::make_shared<ov::opset8::Result>(node_output);
                result_node->set_friendly_name(model_output_name);
                results.push_back(result_node);
            }
        } else if (port_type == "out") {
            const auto& node_outputs = ng_op_map[operation_name];
            FRONT_END_GENERAL_CHECK(node_outputs.size() > port_index,
                                    "Output port with index " + std::to_string(port_index) + " of " + operation_name +
                                        "node specified as custom output does not exist");
            auto result_node = std::make_shared<ov::opset8::Result>(node_outputs[port_index]);
            result_node->set_friendly_name(model_output_name);
            results.push_back(result_node);
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
            size_t producer_port_idx;
            try {
                operation_decoder->get_input_node(port_index, producer_name, producer_port_idx);
            } catch (const std::exception&) {
                FRONT_END_THROW("[ ERROR ] Exception happened when preparing input " + std::to_string(port_index) +
                                " for op '" + operation_decoder->get_op_name() + "', expected input name: '" +
                                producer_name + "', expected input port index: " + std::to_string(producer_port_idx) +
                                '\n');
            }

            // add Result node for this producer output port
            const auto& node_outputs = ng_op_map[producer_name];
            FRONT_END_GENERAL_CHECK(node_outputs.size() > producer_port_idx,
                                    "Output port with index " + std::to_string(producer_port_idx) + " of " +
                                        producer_name + "node specified as custom output does not exist");
            auto result_node = std::make_shared<ov::opset8::Result>(node_outputs[producer_port_idx]);
            result_node->set_friendly_name(model_output_name);
            results.push_back(result_node);
        }
    }
    // find all terminal nodes in OV graph to complete list of results
    if (results.empty()) {
        for (const auto& node_output_vector : ng_op_map) {
            for (size_t output_ind = 0; output_ind < node_output_vector.second.size(); ++output_ind) {
                auto output = node_output_vector.second[output_ind];
                if (output.get_target_inputs().empty() &&
                    !std::dynamic_pointer_cast<ov::opset8::Result>(output.get_node_shared_ptr())) {
                    auto model_output_name =
                        output.get_node_shared_ptr()->get_friendly_name() + ":" + std::to_string(output_ind);
                    auto result_node = std::make_shared<ov::opset8::Result>(output);
                    result_node->set_friendly_name(model_output_name);
                    results.push_back(result_node);
                }
            }
        }
    }

    // TODO: reorder results and params according to indices given in RT info (if any)

    // create the OV Model
    ng_function = std::make_shared<ov::Model>(results, params, model_name);
}

/// \brief Check if FrontEndTensorflow can recognize model from given parts
bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    // TODO: Support other TensorFlow formats: SavedModel, .meta, checkpoint, pbtxt
    if (variants.size() != 1)
        return false;

    // Validating first path, it must contain a model
    if (variants[0].is<std::string>()) {
        std::string suffix = ".pb";
        std::string model_path = variants[0].as<std::string>();
        if (ov::util::ends_with(model_path, suffix.c_str())) {
            return true;
        }
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    else if (variants[0].is<std::wstring>()) {
        std::wstring suffix = L".pb";
        std::wstring model_path = variants[0].as<std::wstring>();
        if (ov::util::ends_with(model_path, suffix)) {
            return true;
        }
    }
#endif
    else if (variants[0].is<GraphIterator::Ptr>()) {
        return true;
    }
    return false;
}

ov::frontend::InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    // TODO: Support other TensorFlow formats: SavedModel, .meta, checkpoint, pbtxt
    if (variants.size() == 1) {
        // a case when binary protobuf format is provided
        if (variants[0].is<std::string>()) {
            std::string suffix = ".pb";
            std::string model_path = variants[0].as<std::string>();
            if (ov::util::ends_with(model_path, suffix.c_str())) {
                return std::make_shared<InputModel>(
                    std::make_shared<::ov::frontend::tensorflow::GraphIteratorProto>(model_path),
                    m_telemetry);
            }
        }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        else if (variants[0].is<std::wstring>()) {
            std::wstring suffix = L".pb";
            std::wstring model_path = variants[0].as<std::wstring>();
            if (ov::util::ends_with(model_path, suffix)) {
                return std::make_shared<InputModel>(
                    std::make_shared<::ov::frontend::tensorflow::GraphIteratorProto>(model_path),
                    m_telemetry);
            }
        }
#endif
        else if (variants[0].is<GraphIterator::Ptr>()) {
            auto graph_iterator = variants[0].as<GraphIterator::Ptr>();
            return std::make_shared<InputModel>(graph_iterator, m_telemetry);
        }
    }
    return nullptr;
}

std::shared_ptr<ov::Model> FrontEnd::convert(const ov::frontend::InputModel::Ptr& model) const {
    auto model_tf = std::dynamic_pointer_cast<InputModel>(model);
    FRONT_END_GENERAL_CHECK(model_tf != nullptr, "Invalid input model");

    if (!m_transformation_extensions.empty()) {
        auto function = decode(model);

        ov::pass::Manager manager;
        for (const auto& transformation : m_transformation_extensions) {
            transformation->register_pass(manager);
        }
        manager.run_passes(function);
        convert(function);
        return function;
    }

    std::shared_ptr<ov::Model> f;
    translate_graph(model_tf, "TensorFlow_Frontend_IR", true, false, f);
    normalize(f);

    for (const auto& node : f->get_ordered_ops()) {
        if (const auto& fw_node = ov::as_type_ptr<ov::frontend::tensorflow::FrameworkNode>(node)) {
            auto op_type = fw_node->get_decoder()->get_op_type();
            auto op_name = fw_node->get_decoder()->get_op_name();
            FRONT_END_OP_CONVERSION_CHECK(
                false,
                "The translation is incomplete due to operation " + op_name + " of type " + op_type);
        }
    }

    return f;
}

std::shared_ptr<ov::Model> FrontEnd::convert_partially(const ov::frontend::InputModel::Ptr& model) const {
    auto model_tf = std::dynamic_pointer_cast<InputModel>(model);
    FRONT_END_GENERAL_CHECK(model_tf != nullptr, "Invalid input model");

    if (!m_transformation_extensions.empty()) {
        auto function = decode(model);

        ov::pass::Manager manager;
        for (const auto& transformation : m_transformation_extensions) {
            transformation->register_pass(manager);
        }
        manager.run_passes(function);
        convert(function);
        return function;
    }

    std::shared_ptr<ov::Model> f;
    translate_graph(model_tf, "TensorFlow_Frontend_IR", false, false, f);
    normalize(f);
    return f;
}

std::shared_ptr<ov::Model> FrontEnd::decode(const ov::frontend::InputModel::Ptr& model) const {
    auto model_tf = std::dynamic_pointer_cast<InputModel>(model);
    std::shared_ptr<ov::Model> f;
    translate_graph(model_tf, "TensorFlow_Frontend_IR", false, true, f);
    return f;
}

void FrontEnd::convert(const std::shared_ptr<ov::Model>& partiallyConverted) const {
    for (const auto& node : partiallyConverted->get_ordered_ops()) {
        if (ov::is_type<FrameworkNode>(node)) {
            translate_framework_node(std::dynamic_pointer_cast<FrameworkNode>(node), m_op_translators);
        }
    }
    for (const auto& result : partiallyConverted->get_results()) {
        result->validate_and_infer_types();
    }

    normalize(partiallyConverted);
}

void FrontEnd::normalize(const std::shared_ptr<ov::Model>& function) const {
    ov::pass::Manager manager;

    // Runs middle transformations to convert sub-graphs with intermediate (frontend internal) operations
    // into sub-graphs with only OpenVINO operations
    manager.register_pass<ov::frontend::tensorflow::pass::EmbeddingSegmentSingleFeatureFusion>();
    manager.register_pass<ov::frontend::tensorflow::pass::BlockLSTMReplacer>();
    manager.register_pass<ov::frontend::tensorflow::pass::GRUBlockCellReplacer>();

    // TODO: reimplement TransposeSinking that does not corrupt filters for Convolution
    // and preserve tensor names in case of sinking
    // manager.register_pass<ov::frontend::tensorflow::pass::TransposeSinking>();
    manager.run_passes(function);
}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    if (auto telemetry = std::dynamic_pointer_cast<TelemetryExtension>(extension)) {
        m_telemetry = telemetry;
    } else if (auto transformation = std::dynamic_pointer_cast<DecoderTransformationExtension>(extension)) {
        m_transformation_extensions.push_back(transformation);
    } else if (const auto& so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(extension)) {
        add_extension(so_ext->extension());
        m_extensions.push_back(so_ext);
    } else if (auto common_conv_ext = std::dynamic_pointer_cast<ov::frontend::ConversionExtension>(extension)) {
        m_conversion_extensions.push_back(common_conv_ext);
        m_op_translators[common_conv_ext->get_op_type()] = [=](const NodeContext& context) {
            return common_conv_ext->get_converter()(context);
        };
    } else if (const auto& tensorflow_conv_ext = std::dynamic_pointer_cast<ConversionExtension>(extension)) {
        m_conversion_extensions.push_back(tensorflow_conv_ext);
        m_op_translators[tensorflow_conv_ext->get_op_type()] = [=](const NodeContext& context) {
            return tensorflow_conv_ext->get_converter()(context);
        };
    }
}
