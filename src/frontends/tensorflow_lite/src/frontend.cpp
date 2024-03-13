// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/tensorflow_lite/frontend.hpp"

#include "graph_iterator_flatbuffer.hpp"
#include "input_model.hpp"
#include "op/op_translation_utils.hpp"
#include "op_table.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/frontend/tensorflow_lite/extension/op.hpp"
#include "openvino/util/common_util.hpp"
#include "tensor_lite_place.hpp"
#include "tf_framework_node.hpp"
#include "tflite_transformations/rfft2d_complex_abs.h"
#include "tflite_transformations/tflite_quantize_resolver.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/resolve_names_collisions.hpp"
#include "transformations/transpose_sinking/ts_general.hpp"

using namespace ov;
using namespace ov::frontend::tensorflow_lite;

namespace {
void translate_framework_node(const std::shared_ptr<ov::frontend::tensorflow::FrameworkNode>& node,
                              const ov::frontend::tensorflow_lite::TranslatorDictionaryType& op_translators) {
    auto type = node->get_op_type();
    const auto& TRANSLATE_OP_MAP = op_translators;
    auto translator_it = TRANSLATE_OP_MAP.find(type);
    FRONT_END_OP_CONVERSION_CHECK(translator_it != TRANSLATE_OP_MAP.end(), "No translator found for ", type, " node.");
    ov::OutputVector ov_inputs = node->input_values();
    ov::frontend::tensorflow_lite::NodeContext node_ctx(node->get_decoder(), ov_inputs);
    auto new_outputs = translator_it->second(node_ctx);
    ov::frontend::tensorflow_lite::op::set_output_names(node_ctx, new_outputs);
    auto old_outputs = node->outputs();
    FRONT_END_GENERAL_CHECK(new_outputs.size() == old_outputs.size());
    for (size_t i = 0; i < new_outputs.size(); ++i) {
        old_outputs[i].replace(new_outputs[i]);
        apply_quantization(new_outputs[i], node->get_output_element_type(i));
    }
}
}  // namespace

FrontEnd::FrontEnd() {
    m_op_translators = ov::frontend::tensorflow_lite::op::get_supported_ops();
}

/// \brief Check if FrontEndTensorflowLite can recognize model from given parts
bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    // Last boolean flag in `variants` (if presented) is reserved for FE configuration
    size_t extra_variants_num = variants.size() > 0 && variants[variants.size() - 1].is<bool>() ? 1 : 0;
    if (variants.size() != 1 + extra_variants_num)
        return false;

    if (variants[0].is<std::string>()) {
        std::string model_path = variants[0].as<std::string>();
        if (GraphIteratorFlatBuffer::is_supported(model_path)) {
            return true;
        }
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    else if (variants[0].is<std::wstring>()) {
        std::wstring model_path = variants[0].as<std::wstring>();
        if (GraphIteratorFlatBuffer::is_supported(model_path)) {
            return true;
        }
    }
#endif
    return false;
}

ov::frontend::InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    // Last boolean flag in `variants` (if presented) is reserved for FE configuration
    size_t extra_variants_num = variants.size() > 0 && variants[variants.size() - 1].is<bool>() ? 1 : 0;
    if (variants.size() == 1 + extra_variants_num) {
        if (variants[0].is<std::string>()) {
            std::string model_path = variants[0].as<std::string>();
            if (GraphIteratorFlatBuffer::is_supported(model_path)) {
                return std::make_shared<tensorflow_lite::InputModel>(
                    std::make_shared<GraphIteratorFlatBuffer>(model_path),
                    m_telemetry);
            }
        }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        else if (variants[0].is<std::wstring>()) {
            std::wstring model_path = variants[0].as<std::wstring>();
            if (GraphIteratorFlatBuffer::is_supported(model_path)) {
                return std::make_shared<tensorflow_lite::InputModel>(
                    std::make_shared<GraphIteratorFlatBuffer>(model_path),
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
        convert(ov_model);
        return ov_model;
    }

    translate_graph(model, true, false, ov_model);
    normalize(ov_model);

    for (const auto& node : ov_model->get_ordered_ops()) {
        if (const auto& fw_node = ov::as_type_ptr<ov::frontend::tensorflow::FrameworkNode>(node)) {
            auto op_type = fw_node->get_decoder()->get_op_type();
            auto op_name = fw_node->get_decoder()->get_op_name();
            FRONT_END_OP_CONVERSION_CHECK(false,
                                          "The translation is incomplete due to operation ",
                                          op_name,
                                          " of type ",
                                          op_type);
        }
    }
    return ov_model;
}

void FrontEnd::convert(const std::shared_ptr<ov::Model>& partiallyConverted) const {
    for (const auto& node : partiallyConverted->get_ordered_ops()) {
        if (ov::is_type<ov::frontend::tensorflow::FrameworkNode>(node)) {
            translate_framework_node(std::dynamic_pointer_cast<ov::frontend::tensorflow::FrameworkNode>(node),
                                     m_op_translators);
        }
    }
    for (const auto& result : partiallyConverted->get_results()) {
        result->validate_and_infer_types();
    }
    normalize(partiallyConverted);
}

std::shared_ptr<ov::Model> FrontEnd::convert_partially(const ov::frontend::InputModel::Ptr& model) const {
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
    translate_graph(model, false, false, f);
    normalize(f);
    return f;
}

void FrontEnd::translate_graph(const InputModel::Ptr& model,
                               bool fail_fast,
                               bool no_conversion,
                               std::shared_ptr<ov::Model>& ov_function) const {
    const auto& model_lite = std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::InputModel>(model);
    FRONT_END_GENERAL_CHECK(model_lite, "nullptr for InputModel is given for translation into OV Model");

    auto subgraphs_as_input_models = model_lite->get_subgraphs();
    auto input_to_ov_model = [&](const std::shared_ptr<ov::frontend::tensorflow_lite::InputModel>& in_model) {
        auto simple_lambda = [&]() -> std::shared_ptr<ov::Model> {
            std::shared_ptr<ov::Model> model;
            if (in_model)
                translate_graph(in_model, fail_fast, no_conversion, model);
            return model;
        };
        return simple_lambda;
    };
    std::vector<std::function<std::shared_ptr<ov::Model>()>> submodel_translation_functions;
    submodel_translation_functions.reserve(subgraphs_as_input_models.size());
    for (const auto& subgraph : subgraphs_as_input_models) {
        submodel_translation_functions.push_back(input_to_ov_model(subgraph));
    }

    const auto& translate_map =
        no_conversion ? ov::frontend::tensorflow_lite::TranslatorDictionaryType{} : m_op_translators;

    auto all_tensor_values = model_lite->get_tensor_values();
    auto all_tensor_places = model_lite->get_tensor_places();

    for (auto& value : all_tensor_values) {
        auto& output = value.second;
        FRONT_END_GENERAL_CHECK(ov::is_type<ov::opset1::Constant>(output.get_node_shared_ptr()),
                                "Unexpected constant data configuration at the beginning of graph translation");
        const auto& input_tensor = all_tensor_places.at(value.first);
        FRONT_END_GENERAL_CHECK(input_tensor != nullptr, "Inputs must be TensorPlaces");
        input_tensor->translate(output, !no_conversion);
    }

    // inputs
    ParameterVector parameters;
    parameters.reserve(model_lite->get_inputs().size());
    for (const auto& input : model_lite->get_inputs()) {
        const auto& input_tensor = std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::TensorLitePlace>(input);
        FRONT_END_GENERAL_CHECK(
            input_tensor != nullptr,
            "Inputs of ov::frontend::tensorflow_lite::InputModel must be TensorLitePlace instances");
        const auto name = input_tensor->get_names()[0];
        auto parameter = std::make_shared<ov::opset1::Parameter>(input_tensor->get_element_type(),
                                                                 input_tensor->get_partial_shape());
        parameter->set_friendly_name(name);
        parameters.push_back(parameter);
        all_tensor_values[name] = parameter->output(0);
        input_tensor->translate(all_tensor_values[name], !no_conversion);
    }

    // operations
    for (const auto& op_place : model_lite->get_op_places()) {
        const auto& decoder = std::dynamic_pointer_cast<tensorflow_lite::DecoderFlatBuffer>(op_place->get_decoder());
        FRONT_END_GENERAL_CHECK(decoder != nullptr, "Decoder must be DecoderFlatBuffer or its child");
        ov::OutputVector inputs(decoder->get_input_size());
        for (size_t i = 0; i < decoder->get_input_size(); ++i) {
            auto name = decoder->get_input_tensor_name(i);
            FRONT_END_GENERAL_CHECK(all_tensor_values.find(name) != all_tensor_values.end(),
                                    "Unknown tensor name: ",
                                    name,
                                    ".");
            inputs[i] = all_tensor_values[name];
        }

        const auto& out_size = decoder->get_output_size();
        ov::OutputVector ov_outputs(out_size);
        try {
            FRONT_END_OP_CONVERSION_CHECK(translate_map.count(decoder->get_op_type()),
                                          "No translator found for " + decoder->get_op_type() + " node.");
            auto op_fun = &(translate_map.at(decoder->get_op_type()));
            ov::frontend::tensorflow_lite::NodeContext node_context(decoder, inputs, submodel_translation_functions);
            ov_outputs = (*op_fun)(node_context);
        } catch (...) {
            if (fail_fast) {
                if (m_telemetry && translate_map.count(decoder->get_op_type()) == 0) {
                    m_telemetry->send_event("error_cause", "tflite_" + decoder->get_op_type());
                }
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
            all_tensor_places[name]->translate(all_tensor_values[name], !no_conversion);
        }
    }

    // outputs
    ResultVector results;
    results.reserve(model_lite->get_outputs().size());
    for (const auto& output : model_lite->get_outputs()) {
        const auto& tensor = std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::TensorLitePlace>(output);
        FRONT_END_GENERAL_CHECK(
            tensor != nullptr,
            "Inputs of ov::frontend::tensorflow_lite::InputModel must be TensorLitePlace instances");
        const auto name = tensor->get_names()[0];
        if (!all_tensor_values.count(name)) {
            continue;
        }
        const auto& output_value = all_tensor_values[name];
        const auto& result = std::make_shared<ov::opset1::Result>(output_value);
        auto input = result->output(0);
        tensor->translate(input, !no_conversion);
        results.push_back(result);
    }
    auto model_name = "TensorFlow_Lite_Frontend_IR";
    ov_function = std::make_shared<ov::Model>(results, parameters, model_name);
}

std::shared_ptr<ov::Model> FrontEnd::decode(const InputModel::Ptr& model) const {
    std::shared_ptr<ov::Model> ov_model;
    translate_graph(model, false, true, ov_model);
    return ov_model;
}

void FrontEnd::normalize(const std::shared_ptr<ov::Model>& function) const {
    ov::pass::Manager manager;
    // Mark quantized and f16/bf16 compressed constants to prevent CF for them,
    // so that not extra memory is used for intermediate decompressed constants.
    manager.register_pass<ov::pass::MarkCompressedFloatConstants>();
    manager.register_pass<ov::frontend::tensorflow_lite::pass::TFLQuantizeResolver>();
    manager.register_pass<ov::frontend::tensorflow_lite::pass::Rfft2dSimplifier>();
    manager.register_pass<ov::pass::TransposeSinking>();
    manager.register_pass<ov::pass::TransposeSinkingGeneral>();
    manager.register_pass<ov::pass::ResolveNameCollisions>();
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
    } else if (const auto& tensorflow_conv_ext =
                   std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::ConversionExtension>(extension)) {
        m_conversion_extensions.push_back(tensorflow_conv_ext);
        m_op_translators[tensorflow_conv_ext->get_op_type()] = [=](const NodeContext& context) {
            return tensorflow_conv_ext->get_converter()(context);
        };
    } else if (auto op_base_ext = std::dynamic_pointer_cast<ov::BaseOpExtension>(extension)) {
        for (const auto& attached_ext : op_base_ext->get_attached_extensions()) {
            add_extension(attached_ext);
        }
    }
}
