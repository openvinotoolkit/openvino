// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/tensorflow_lite/frontend.hpp"
#include "input_model.hpp"
#include "lite_op_table.hpp"
#include "tf_framework_node.hpp"
#include "graph_iterator_flatbuffer.hpp"
#include "openvino/util/common_util.hpp"
#include "pass/transpose_sinking.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "tensor_lite_place.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "op/op_translation_utils.hpp"

using namespace ov;
using namespace ov::frontend::tensorflow_lite;

FrontEnd::FrontEnd() {
    m_op_translators = ov::frontend::tensorflow_lite::op::get_supported_ops();
}

/// \brief Check if FrontEndTensorflowLite can recognize model from given parts
bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    if (variants.size() != 1)
        return false;

    if (variants[0].is<std::string>()) {
        std::string suffix = ".tflite";
        std::string model_path = variants[0].as<std::string>();
        if (ov::util::ends_with(model_path, suffix.c_str())) {
            return true;
        }
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    else if (variants[0].is<std::wstring>()) {
        std::wstring suffix = L".tflite";
        std::wstring model_path = variants[0].as<std::wstring>();
        if (ov::util::ends_with(model_path, suffix)) {
            return true;
        }
    }
#endif
    return false;
}

ov::frontend::InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    if (variants.size() == 1) {
        if (variants[0].is<std::string>()) {
            std::string suffix = ".tflite";
            std::string model_path = variants[0].as<std::string>();
            if (ov::util::ends_with(model_path, suffix.c_str())) {
                return std::make_shared<tensorflow_lite::InputModel>(std::make_shared<GraphIteratorFlatBuffer>(model_path),
                                                                     m_telemetry);
            }
        }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        else if (variants[0].is<std::wstring>()) {
            std::wstring suffix = L".tflite";
            std::wstring model_path = variants[0].as<std::wstring>();
            if (ov::util::ends_with(model_path, suffix)) {
                return std::make_shared<InputModel>(
                    std::make_shared<::ov::frontend::tensorflow::GraphIteratorFlatBuffer>(model_path),
                    m_telemetry);
            }
        }
#endif
    }
    return nullptr;
}

std::shared_ptr<ov::Model> FrontEnd::convert(const ov::frontend::InputModel::Ptr& model) const {
    std::shared_ptr<ov::Model> ov_model;
    if (!m_transformation_extensions.empty()) {
        auto ov_model = decode(model);

        ov::pass::Manager manager;
        for (const auto& transformation : m_transformation_extensions) {
            transformation->register_pass(manager);
        }
        manager.run_passes(ov_model);
// FIXME       convert(ov_model);
        return ov_model;
    }

    translate_graph(model, "TensorFlow_Lite_Frontend_IR", true, false, ov_model);
    normalize(ov_model);
    ov::pass::VisualizeTree("frontend_ov_model_normalized.svg").run_on_model(ov_model);

    for (const auto& node : ov_model->get_ordered_ops()) {
        if (const auto& fw_node = ov::as_type_ptr<ov::frontend::tensorflow::FrameworkNode>(node)) {
            auto op_type = fw_node->get_decoder()->get_op_type();
            auto op_name = fw_node->get_decoder()->get_op_name();
            FRONT_END_OP_CONVERSION_CHECK(
                    false,
                    "The translation is incomplete due to operation ", op_name, " of type ", op_type);
        }
    }
    return ov_model;
}

void FrontEnd::translate_graph(const InputModel::Ptr &model, const std::string &model_name, bool fail_fast,
                               bool no_conversion, std::shared_ptr<ov::Model>& ov_function) const {
    const auto& model_lite = std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::InputModel>(model);
    FRONT_END_GENERAL_CHECK(model_lite, "nullptr for InputModel is given for translation into OV Model");

    const auto& translate_map = no_conversion ? ov::frontend::tensorflow::TranslatorDictionaryType{} : m_op_translators;

    auto all_tensor_values = model_lite->get_tensor_values();
    auto all_tensor_places = model_lite->get_tensor_places();

    for (auto& value : all_tensor_values) {
        auto output = value.second;
        FRONT_END_GENERAL_CHECK(ov::is_type<ov::opset1::Constant>(output.get_node_shared_ptr()),
                "Unexpected constant data configuration at the beginning of graph translation");
        const auto& input_tensor = std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::TensorLitePlace>(
                all_tensor_places.at(value.first));
        FRONT_END_GENERAL_CHECK(input_tensor != nullptr, "Inputs must be TensorPlaces");
        output.get_node_shared_ptr()->set_friendly_name(*input_tensor->get_names().begin());
        output.set_names({*input_tensor->get_names().begin()});
        value.second = apply_quantization(output, input_tensor);
    }

    // inputs
    ParameterVector parameters;
    parameters.reserve(model_lite->get_inputs().size());
    for (const auto& input : model_lite->get_inputs()) {
        const auto& input_tensor = std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::TensorLitePlace>(input);
        FRONT_END_GENERAL_CHECK(input_tensor != nullptr, "Inputs must be TensorPlaces");
        const auto name = input_tensor->get_names()[0];
        auto parameter = std::make_shared<ov::opset1::Parameter>(
                input_tensor->get_element_type(), input_tensor->get_partial_shape());
        parameter->set_friendly_name(name);
        parameters.push_back(parameter);
        parameter->get_output_tensor(0).set_names({name});
        Output<Node> output = parameter->output(0);
        if (!no_conversion) {
            output = apply_quantization(output, input_tensor, true);
        }
        all_tensor_values[name] = output;
    }

    // outputs
    ResultVector results;
    results.reserve(model_lite->get_outputs().size());
    std::unordered_map<std::string, std::shared_ptr<tensorflow::TensorPlace>> output_names;
    for (const auto& output : model_lite->get_outputs()) {
        const auto& output_tensor = std::dynamic_pointer_cast<tensorflow::TensorPlace>(output);
        FRONT_END_GENERAL_CHECK(output_tensor != nullptr, "Outputs must be TensorPlaces");
        const auto name = output_tensor->get_names()[0];
        output_names[name] = output_tensor;
    }

    // operations
    for (const auto& op_place : model_lite->get_op_places()) {
        const auto& decoder = std::dynamic_pointer_cast<tensorflow_lite::DecoderFlatBuffer>(op_place->get_decoder());
        FRONT_END_GENERAL_CHECK(decoder != nullptr, "Decoder must be DecoderFlatBuffer or its child");
        ov::OutputVector inputs(decoder->get_input_size());
        for (size_t i = 0; i < decoder->get_input_size(); ++i) {
            size_t tensor_idx;
            std::string tensor_name;
            decoder->get_input_node(i, tensor_name, tensor_idx);
            ov::Output<Node> input;
            FRONT_END_GENERAL_CHECK(all_tensor_values.find(tensor_name) != all_tensor_values.end(), "Unknown tensor name: ", tensor_name, ".");
            inputs[i] = all_tensor_values[tensor_name];
        }

        const auto& out_size = decoder->get_output_size();

        ov::OutputVector ov_outputs(out_size);
        try {
            FRONT_END_OP_CONVERSION_CHECK(translate_map.count(decoder->get_op_type()),
                                          "No translator found for " + decoder->get_op_type() + " node.");
            auto op_fun = &(translate_map.at(decoder->get_op_type()));
            ov::frontend::tensorflow::NodeContext node_context(decoder, inputs);
            ov_outputs = (*op_fun)(node_context);
            std::cout << ov_outputs[0].get_node_shared_ptr() << std::endl;
            op::set_output_names(node_context, ov_outputs);
        } catch (...) {
            if (fail_fast) {
                // re-throw any exception
                throw;
            } else {
                auto operation = std::make_shared<ov::frontend::tensorflow::FrameworkNode>(decoder, inputs, out_size);
                operation->set_friendly_name(decoder->get_op_name());
                ov_outputs = operation->outputs();
            }
        }
        for (size_t i = 0; i < out_size; ++i) {
            const auto& name = decoder->get_output_tensor_name(i);
            all_tensor_values[name] = ov_outputs[i];
            ov_outputs[i].get_tensor().set_names({name});
            if (!no_conversion) {
                ov_outputs[i] = apply_quantization(ov_outputs[i], all_tensor_places[name]);
            }
            if (output_names.find(name) != output_names.end()) {
                const auto& result = std::make_shared<ov::opset1::Result>(ov_outputs[i]); // order may be different!
                result->set_friendly_name(name);
                result->get_output_tensor(i).set_names({name});
                results.push_back(result);
            }
        }
    }
    ov_function = std::make_shared<ov::Model>(results, parameters, model_name);
}

std::shared_ptr<ov::Model> FrontEnd::decode(const InputModel::Ptr &model) const {
    std::shared_ptr<ov::Model> ov_model;
    translate_graph(model, "TensorFlow_Lite_Frontend_IR", false, true, ov_model);
    return ov_model;
}

void FrontEnd::normalize(const std::shared_ptr<ov::Model> &function) const {
    ov::pass::Manager manager;
    // TODO: register i8 weights normalization after implemented
    // TODO: remove custom transpose sinking after common TS ready
    manager.register_pass<ov::pass::TransposeSinking>();
    manager.register_pass<ov::frontend::tensorflow::pass::TransposeSinking>();
    manager.run_passes(function);
}
