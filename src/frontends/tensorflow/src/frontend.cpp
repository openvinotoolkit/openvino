// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/tensorflow/frontend.hpp"

#include "graph_iterator_proto.hpp"
#include "helper_transforms/block_lstm_replacer.hpp"
#include "helper_transforms/embedding_segments_feature_fusing.hpp"
#include "helper_transforms/gru_block_cell_replacer.hpp"
#include "helper_transforms/structural_type_prop.hpp"
#include "input_model.hpp"
#include "op_table.hpp"
#include "openvino/frontend/tensorflow/extension/conversion.hpp"
#include "openvino/frontend/tensorflow/graph_iterator.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "pass/transpose_sinking.hpp"
#include "so_extension.hpp"
#include "tf_framework_node.hpp"
#include "transformations/common_optimizations/reverse_shape_and_type_infer.hpp"
#include "translate_session.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::frontend::tensorflow;

namespace {
void translate_framework_node(const std::shared_ptr<FrameworkNode>& node,
                              const TranslatorDictionaryType& op_translators) {
    auto type = node->get_op_type();

    const auto& TRANSLATE_OP_MAP = op_translators;
    auto translator_it = TRANSLATE_OP_MAP.find(type);
    FRONT_END_OP_CONVERSION_CHECK(translator_it != TRANSLATE_OP_MAP.end(), "No translator found for ", type, " node.");

    ov::OutputVector ov_inputs = node->input_values();
    NodeContext node_ctx(node->get_decoder(), ov_inputs);
    auto new_node_outputs = translator_it->second(node_ctx);

    auto new_output = new_node_outputs.begin();
    auto old_outputs = node->outputs();
    auto old_output = old_outputs.begin();

    for (; new_output != new_node_outputs.end() && old_output != old_outputs.end(); ++old_output, ++new_output) {
        old_output->replace(*new_output);
    }
}
}  // namespace



FrontEnd::FrontEnd() : m_op_translators(tensorflow::op::get_supported_ops()) { std::cerr << "[ INFO ] TensorFlow FE is initialized\n"; }


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

    // create a shared pointer to the cloned dictionary of translators
    auto translator_map = std::make_shared<TranslatorDictionaryType>(m_op_translators);

    std::shared_ptr<ov::Model> f;
    TranslateSession translate_session(model, translator_map, "TensorFlow_Frontend_IR", true, m_telemetry != nullptr);
    try {
        f = translate_session.get_converted_model();
    } catch (const std::exception&) {
        if (m_telemetry) {
            auto telemetry_data = translate_session.get_telemetry_data();
            if (telemetry_data) {
                // send event about which operation is not supported for conversion
                for (const auto& telemetry_item : *telemetry_data.get()) {
                    m_telemetry->send_event(telemetry_item.first, telemetry_item.second);
                }
            }
        }
        throw;
    }
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

        std::cerr << "[ INFO ] About to start decoding\n";

        auto function = decode(model);

        std::cerr << "[ INFO ] Decoded\n";

        ov::pass::Manager manager;
        for (const auto& transformation : m_transformation_extensions) {
            transformation->register_pass(manager);
        }
        std::cerr << "[ INFO ] About to start passes\n";
        manager.run_passes(function);
        std::cerr << "[ INFO ] Passes are completed, start conversion\n";
        convert(function);  // ERROR! It will throw an exception if there is at least one unsupported operation, but we are in convert_partially that shouldn't throw in this case
        std::cerr << "[ INFO ] Converted\n";
        return function;
    }

    // create a shared pointer to the cloned dictionary of translators
    auto translator_map = std::make_shared<TranslatorDictionaryType>(m_op_translators);

    std::shared_ptr<ov::Model> f;
    TranslateSession translate_session(model, translator_map, "TensorFlow_Frontend_IR", false, m_telemetry != nullptr);
    try {
        f = translate_session.get_converted_model();
    } catch (const std::exception&) {
        if (m_telemetry) {
            auto telemetry_data = translate_session.get_telemetry_data();
            if (telemetry_data) {
                // send event about which operation is not supported for conversion
                for (const auto& telemetry_item : *telemetry_data.get()) {
                    m_telemetry->send_event(telemetry_item.first, telemetry_item.second);
                }
            }
        }
        throw;
    }
    normalize(f);

    return f;
}

std::shared_ptr<ov::Model> FrontEnd::decode(const ov::frontend::InputModel::Ptr& model) const {
    auto translator_map = std::make_shared<TranslatorDictionaryType>();

    const std::set<std::string> required_types{"Placeholder", "NoOp"};
    for (const auto& name : required_types) {
        translator_map->emplace(name, m_op_translators.at(name));
    }

    std::shared_ptr<ov::Model> f;
    TranslateSession translate_session(model, translator_map, "TensorFlow_Frontend_IR", false, m_telemetry != nullptr);
    try {
        f = translate_session.get_converted_model();
    } catch (const std::exception&) {
        if (m_telemetry) {
            auto telemetry_data = translate_session.get_telemetry_data();
            if (telemetry_data) {
                // send event about which operation is not supported for conversion
                for (const auto& telemetry_item : *telemetry_data.get()) {
                    m_telemetry->send_event(telemetry_item.first, telemetry_item.second);
                }
            }
        }
        throw;
    }

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
    manager.register_pass<pass::EmbeddingSegmentSingleFeatureFusion>();
    manager.register_pass<pass::BlockLSTMReplacer>();
    manager.register_pass<pass::GRUBlockCellReplacer>();
    manager.set_per_pass_validation(true);
    //manager.register_pass<ov::pass::GraphRewrite>(std::make_shared<pass::StructuralTypeProp>());
    //manager.register_pass<ov::pass::GraphRewrite>(std::make_shared<pass::ReplaceStrByU81D>());
    manager.register_pass<pass::ReplaceParameterByVocab>();
    manager.register_pass<pass::DecomposeStrParameters>();
    auto propagators = manager.register_pass<ov::pass::GraphRewrite>();
    propagators->add_matcher<ov::pass::GraphRewrite>(std::make_shared<pass::ThroughStrOpsProp>());
    propagators->add_matcher<ov::pass::GraphRewrite>(std::make_shared<pass::ThroughReshapeProp>());
    propagators->add_matcher<ov::pass::GraphRewrite>(std::make_shared<pass::ThroughNotEqualProp>());
    propagators->add_matcher<ov::pass::GraphRewrite>(std::make_shared<pass::ThroughWhileProp>());
    propagators->add_matcher<ov::pass::GraphRewrite>(std::make_shared<pass::ThroughTensorListStack>());
    manager.register_pass<pass::DecomposeStructResults>();

    manager.set_per_pass_validation(false);

    // TODO: reimplement TransposeSinking that does not corrupt filters for Convolution
    manager.register_pass<ov::frontend::tensorflow::pass::TransposeSinking>();
    manager.register_pass<ov::pass::ReverseShapeAndTypeInfer>();
    manager.run_passes(function);
    std::cerr << "[ END OF TRANSFORMATIONS ]\n";
    //function->validate_nodes_and_infer_types();
    //std::cerr << "End of validation\n";
    //serialize(function, "function.xml");
    //std::cerr << "End of serialization of function\n";
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
