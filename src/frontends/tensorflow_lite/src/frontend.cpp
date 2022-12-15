// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/tensorflow_lite/frontend.hpp"

#include "graph_iterator_flatbuffer.hpp"
#include "input_model.hpp"
#include "op_table.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/visualize_tree.hpp"

using namespace ov;
using namespace ov::frontend::tensorflow_lite;

FrontEnd::FrontEnd() {
    m_op_translators = tensorflow::op::get_supported_lite_ops();
}

/// \brief Check if FrontEndTensorflow can recognize model from given parts
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

std::shared_ptr<ov::Model> FrontEnd::convert(const ov::frontend::InputModel::Ptr &model) const {
    auto model_tf = std::dynamic_pointer_cast<InputModel>(model);
    FRONT_END_GENERAL_CHECK(model_tf != nullptr, "Invalid input model");
    std::shared_ptr<ov::Model> ov_model;
    translate_graph(model, "TF Lite Frontend IR", true, false, ov_model);
    return ov_model;
}

void FrontEnd::translate_graph(const InputModel::Ptr &model, const std::string &model_name, bool fail_fast,
                               bool no_conversion, std::shared_ptr<ov::Model> &ng_function) const {
    const auto& model_lite = std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::InputModel>(model);
    FRONT_END_GENERAL_CHECK(model_lite, "nullptr for InputModel is given for translation into OV Model");

    auto all_tensor_values = model_lite->get_tensor_values();

    // inputs
    ParameterVector parameters;
    parameters.reserve(model_lite->get_inputs().size());
    for (const auto& input : model_lite->get_inputs()) {
        const auto& input_tensor = std::dynamic_pointer_cast<tensorflow::TensorPlace>(input);
        FRONT_END_GENERAL_CHECK(input_tensor != nullptr, "Inputs must be TensorPlaces");
        const auto name = input_tensor->get_names()[0];
        const auto& parameter = std::make_shared<ov::opset1::Parameter>(
                input_tensor->get_element_type(), input_tensor->get_partial_shape());
        parameter->set_friendly_name(name);
        parameters.push_back(parameter);
        all_tensor_values[name] = parameter->output(0);
        parameter->get_output_tensor(0).set_names({name});
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
        const auto& operation = std::make_shared<op::util::FrameworkNode>(inputs, out_size);
        operation->set_friendly_name(decoder->get_op_name());
        for (size_t i = 0; i < out_size; ++i) {
            const auto& name = decoder->get_output_tensor_name(i);
            all_tensor_values[name] = operation->output(i);
            operation->get_output_tensor(i).set_names({name});
            if (output_names.find(name) != output_names.end()) {
                const auto& result = std::make_shared<ov::opset1::Result>(operation->output(i));
                result->set_friendly_name(name);
                result->get_output_tensor(i).set_names({name});
                results.push_back(result);
            }
        }
    }
    ng_function = std::make_shared<ov::Model>(results, parameters, model_name);
}
