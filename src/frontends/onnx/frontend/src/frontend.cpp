// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <google/protobuf/port_def.inc>
#ifndef PROTOBUF_VERSION
#    include <google/protobuf/runtime_version.h>
#endif
#if PROTOBUF_VERSION >= 4022000  // protobuf 4.22
#    define OV_PROTOBUF_ABSL_IS_USED
#endif
#include <google/protobuf/port_undef.inc>

#ifdef OV_PROTOBUF_ABSL_IS_USED
#    include <absl/log/globals.h>
#else
#    include <google/protobuf/stubs/logging.h>
#endif

#include <fstream>
#include <sstream>

#include "input_model.hpp"
#include "onnx_common/onnx_model_validator.hpp"
#include "onnx_framework_node.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/onnx/extension/conversion.hpp"
#include "openvino/frontend/onnx/frontend.hpp"
#include "openvino/frontend/onnx/visibility.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "ops_bridge.hpp"
#include "transformations/resolve_names_collisions.hpp"
#include "utils/common.hpp"
#include "utils/onnx_internal.hpp"

using namespace ov;
using namespace ov::frontend::onnx;
using namespace ov::frontend::onnx::common;
using ::ONNX_NAMESPACE::ModelProto;
using ::ONNX_NAMESPACE::Version;

namespace {
// !!! Experimental feature, it may be changed or removed in the future !!!
void enumerate_constants(const std::shared_ptr<ov::Model>& model) {
    const auto& operations = model->get_ordered_ops();
    for (uint32_t idx = 0; idx < operations.size(); ++idx) {
        const auto& const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(operations[idx]);
        if (const_node == nullptr)
            continue;
        const_node->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
            ov::WeightlessCacheAttribute(0, idx, const_node->get_element_type());
    }
}
// !!! End of Experimental feature
}  // namespace

ONNX_FRONTEND_C_API ov::frontend::FrontEndVersion get_api_version() {
    return OV_FRONTEND_API_VERSION;
}

ONNX_FRONTEND_C_API void* get_front_end_data() {
    ov::frontend::FrontEndPluginInfo* res = new ov::frontend::FrontEndPluginInfo();
    res->m_name = "onnx";
    res->m_creator = []() {
        return std::make_shared<FrontEnd>();
    };
#ifndef OPENVINO_DEBUG_ENABLE
    // disable protobuf logging
#    ifdef OV_PROTOBUF_ABSL_IS_USED
    absl::SetGlobalVLogLevel(0);
#    else
    google::protobuf::SetLogHandler(nullptr);
#    endif
#endif
    return res;
}

ov::frontend::InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    if (variants.empty()) {
        return nullptr;
    }
    // enable mmap by default
    const bool enable_mmap = variants[variants.size() - 1].is<bool>() ? variants[variants.size() - 1].as<bool>() : true;

    if (variants[0].is<std::string>()) {
        const auto path = variants[0].as<std::string>();
        return std::make_shared<InputModel>(path, enable_mmap, m_extensions);
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    if (variants[0].is<std::wstring>()) {
        const auto path = variants[0].as<std::wstring>();
        return std::make_shared<InputModel>(path, enable_mmap, m_extensions);
    }
#endif
    if (variants[0].is<std::istream*>()) {
        const auto stream = variants[0].as<std::istream*>();
        if (variants.size() > 1 && variants[1].is<std::string>()) {
            const auto path = variants[1].as<std::string>();
            return std::make_shared<InputModel>(*stream, path, enable_mmap, m_extensions);
        }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        if (variants.size() > 1 && variants[1].is<std::wstring>()) {
            const auto path = variants[1].as<std::wstring>();
            return std::make_shared<InputModel>(*stream, path, enable_mmap, m_extensions);
        }
#endif
        return std::make_shared<InputModel>(*stream, enable_mmap, m_extensions);
    }
    // !!! Experimental feature, it may be changed or removed in the future !!!
    if (variants[0].is<uint64_t>()) {
        void* model_proto_addr = reinterpret_cast<void*>(variants[0].as<uint64_t>());
        FRONT_END_GENERAL_CHECK(model_proto_addr != 0, "Wrong address of a ModelProto object is passed");
        ModelProto* model_proto_ptr = static_cast<ModelProto*>(model_proto_addr);
        FRONT_END_GENERAL_CHECK(
            model_proto_ptr->has_ir_version() && model_proto_ptr->ir_version() < Version::IR_VERSION,
            "A ModelProto object contains unsupported IR version");
        return std::make_shared<InputModel>(std::make_shared<ModelProto>(*model_proto_ptr), m_extensions);
    }
    // !!! End of Experimental feature
    if (variants[0].is<GraphIterator::Ptr>()) {
        auto graph_iterator = variants[0].as<GraphIterator::Ptr>();
        return std::make_shared<unify::InputModel>(graph_iterator);
    }
    return nullptr;
}

std::shared_ptr<ov::Model> FrontEnd::convert_partially(const ov::frontend::InputModel::Ptr& input_model) const {
    auto unify_model = std::dynamic_pointer_cast<unify::InputModel>(input_model);
    if (unify_model != nullptr) {
        return convert_partially_unify(unify_model);
    }
    auto model_onnx = std::dynamic_pointer_cast<InputModel>(input_model);
    FRONT_END_GENERAL_CHECK(model_onnx != nullptr, "Invalid input model");

    if (!m_transformation_extensions.empty()) {
        auto model = decode(input_model);

        ov::pass::Manager manager("Frontend:ONNX:convert_partially");
        for (const auto& transformation : m_transformation_extensions) {
            transformation->register_pass(manager);
        }
        manager.run_passes(model);
        convert(model);
        return model;
    }

    const auto& converted_model = model_onnx->convert();

    ov::frontend::onnx::common::collect_translation_exceptions(converted_model, m_extensions.telemetry);

    normalize(converted_model);
    return converted_model;
}

void FrontEnd::normalize(const std::shared_ptr<ov::Model>& model) const {
    // !!! Experimental feature, it may be changed or removed in the future !!!
    enumerate_constants(model);
    // !!! End of Experimental feature

    // Here, you can register transformations as a second step of importing process
    // In particular, you can operate on not supported ops (it allows to N:N ONNX->OV mapping).
    ov::pass::Manager manager("Frontend:ONNX:normalize");
    manager.register_pass<pass::ResolveNameCollisions>(true);
    manager.run_passes(model);
}

std::shared_ptr<ov::Model> FrontEnd::convert(const InputModel::Ptr& input_model) const {
    auto unify_model = std::dynamic_pointer_cast<unify::InputModel>(input_model);
    if (unify_model != nullptr) {
        return convert_unify(unify_model);
    }

    auto model_onnx = std::dynamic_pointer_cast<InputModel>(input_model);
    FRONT_END_GENERAL_CHECK(model_onnx != nullptr, "Invalid input model");

    if (!m_transformation_extensions.empty()) {
        auto model = decode(input_model);

        ov::pass::Manager manager("Frontend:ONNX:convert");
        for (const auto& transformation : m_transformation_extensions) {
            transformation->register_pass(manager);
        }
        manager.run_passes(model);
        convert(model);
        return model;
    }

    const auto& converted_model = model_onnx->convert();

    std::stringstream error_messages;

    if (ov::frontend::onnx::common::collect_translation_exceptions(converted_model,
                                                                   m_extensions.telemetry,
                                                                   &error_messages)) {
        FRONT_END_THROW(error_messages.str());
    }

    normalize(converted_model);
    return converted_model;
}

void FrontEnd::convert(const std::shared_ptr<ov::Model>& partially_converted) const {
    ov::frontend::onnx::detail::convert_decoded_model(partially_converted);
    normalize(partially_converted);
}

std::shared_ptr<ov::Model> FrontEnd::decode(const ov::frontend::InputModel::Ptr& model) const {
    auto unify_model = std::dynamic_pointer_cast<unify::InputModel>(model);
    if (unify_model != nullptr) {
        return decode_unify(unify_model);
    }
    auto model_onnx = std::dynamic_pointer_cast<InputModel>(model);
    FRONT_END_GENERAL_CHECK(model_onnx != nullptr, "Invalid input model");
    return model_onnx->decode();
}

std::string FrontEnd::get_name() const {
    return "onnx";
}

namespace {
/**
 * This helper struct uses RAII to rewind/reset the stream so that it points to the beginning
 * of the underlying resource (string, file, and so on). It works similarly to std::lock_guard,
 * which releases a mutex upon destruction.
 *
 * This ensures that the stream is always reset (exception, successful and unsuccessful
 * model validation).
 */
struct StreamRewinder {
    StreamRewinder(std::istream& stream) : m_stream(stream) {
        m_stream.seekg(0, m_stream.beg);
    }
    ~StreamRewinder() {
        m_stream.seekg(0, m_stream.beg);
    }

private:
    std::istream& m_stream;
};
}  // namespace

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    if (variants.size() == 0) {
        return false;
    }
    std::ifstream model_stream;
    if (variants[0].is<std::string>()) {
        const auto path = variants[0].as<std::string>();
        validate_path(path);
        model_stream.open(path, std::ios::in | std::ifstream::binary);
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    else if (variants[0].is<std::wstring>()) {
        const auto path = variants[0].as<std::wstring>();
        validate_path(path);
        model_stream.open(path.c_str(), std::ios::in | std::ifstream::binary);
    }
#endif
    if (model_stream.is_open()) {
        model_stream.seekg(0, model_stream.beg);
        const bool is_valid_model = ::ov::frontend::onnx::common::is_valid_model(model_stream);
        model_stream.close();
        return is_valid_model;
    }
    if (variants[0].is<std::istream*>()) {
        const auto stream = variants[0].as<std::istream*>();
        StreamRewinder rwd{*stream};
        return is_valid_model(*stream);
    }
    // !!! Experimental feature, it may be changed or removed in the future !!!
    if (variants[0].is<uint64_t>()) {
        void* model_proto_addr = reinterpret_cast<void*>(variants[0].as<uint64_t>());
        if (model_proto_addr == 0) {
            return false;
        }
        ModelProto* model_proto_ptr = static_cast<ModelProto*>(model_proto_addr);
        try {
            if (!model_proto_ptr->has_ir_version() || model_proto_ptr->ir_version() > Version::IR_VERSION) {
                return false;
            }
        } catch (...) {
            return false;
        }
        return true;
    }
    // !!! End of Experimental feature
    if (variants[0].is<GraphIterator::Ptr>()) {
        return true;
    }
    return false;
}

std::shared_ptr<ov::Model> FrontEnd::convert_unify(const InputModel::Ptr& input_model) const {
    std::shared_ptr<ov::Model> ov_model;
    if (!m_transformation_extensions.empty()) {
        auto ov_model = decode(input_model);

        ov::pass::Manager manager("Frontend:TFLite:convert");
        for (const auto& transformation : m_transformation_extensions) {
            transformation->register_pass(manager);
        }
        manager.run_passes(ov_model);
        convert(ov_model);
        return ov_model;
    }

    translate_graph(input_model, true, false, ov_model);
    normalize(ov_model);

    for (const auto& node : ov_model->get_ordered_ops()) {
        if (const auto& fw_node = ov::as_type_ptr<ov::frontend::onnx::ONNXFrameworkNode>(node)) {
            /*
            auto op_type = fw_node->get_decoder()->get_op_type();
            auto op_name = fw_node->get_decoder()->get_op_name();
            FRONT_END_OP_CONVERSION_CHECK(false,
                                          "The translation is incomplete due to operation ",
                                          op_name,
                                          " of type ",
                                          op_type);
                                          */
        }
    }
    return ov_model;
}

std::shared_ptr<ov::Model> FrontEnd::convert_partially_unify(const InputModel::Ptr& input_model) const {
    if (!m_transformation_extensions.empty()) {
        auto function = decode_unify(input_model);
        ov::pass::Manager manager("Frontend:ONNX:convert_partially");
        for (const auto& transformation : m_transformation_extensions) {
            transformation->register_pass(manager);
        }
        manager.run_passes(function);
        convert(function);
        return function;
    }

    std::shared_ptr<ov::Model> f;
    translate_graph(input_model, false, false, f);
    normalize(f);
    return f;
}
void FrontEnd::translate_graph(const InputModel::Ptr& input_model,
                               bool fail_fast,
                               bool no_conversion,
                               std::shared_ptr<ov::Model>& ov_function) const {
    auto model_onnx = std::dynamic_pointer_cast<unify::InputModel>(input_model);
    FRONT_END_GENERAL_CHECK(model_onnx != nullptr, "Invalid input model");
    ov_function = model_onnx->get_model();
    return;
#if 0
    auto subgraphs_as_input_models = model_onnx->get_subgraphs();
    auto input_to_ov_model = [&](const std::shared_ptr<ov::frontend::onnx::unify::InputModel>& in_model) {
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

    const OperatorsBridge translate_map;
    //    no_conversion ? ov::frontend::onnx::TranslatorDictionaryType{} : m_op_translators;

    auto all_tensor_values = model_onnx->get_tensor_values();
    auto all_tensor_places = model_onnx->get_tensor_places();

    for (auto& value : all_tensor_values) {
        auto& output = value.second;
        FRONT_END_GENERAL_CHECK(ov::is_type<ov::op::v0::Constant>(output.get_node_shared_ptr()),
                                "Unexpected constant data configuration at the beginning of graph translation");
        const auto& input_tensor = all_tensor_places.at(value.first);
        FRONT_END_GENERAL_CHECK(input_tensor != nullptr, "Inputs must be TensorPlaces");
        input_tensor->translate(output, !no_conversion);
    }

    // inputs
    ParameterVector parameters;
    parameters.reserve(model_onnx->get_inputs().size());
    for (const auto& input : model_onnx->get_inputs()) {
        const auto& input_tensor = std::dynamic_pointer_cast<ov::frontend::onnx::TensorONNXPlace>(input);
        FRONT_END_GENERAL_CHECK(input_tensor != nullptr,
                                "Inputs of ov::frontend::onnx::InputModel must be TensorLitePlace instances");
        const auto name = input_tensor->get_names()[0];
        auto parameter = std::make_shared<ov::op::v0::Parameter>(input_tensor->get_element_type(),
                                                                 input_tensor->get_partial_shape());
        parameter->set_friendly_name(name);
        parameters.push_back(parameter);
        all_tensor_values[name] = parameter->output(0);
        input_tensor->translate(all_tensor_values[name], !no_conversion);
    }

    // operations
    for (const auto& op_place : model_onnx->get_op_places()) {
        const auto& decoder = std::dynamic_pointer_cast<onnx::DecoderBaseOperation>(op_place->get_decoder());
        FRONT_END_GENERAL_CHECK(decoder != nullptr, "Decoder must be onnx::DecoderBase or its child");
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
        const Operator* translator =
            translate_map.get_operator(decoder->get_domain(), decoder->get_op_type(), decoder->get_op_set());
        try {
            FRONT_END_OP_CONVERSION_CHECK(
                translator != nullptr,
                "No translator found for " + decoder->get_domain() + " " + decoder->get_op_type() + " node.");
            // FRONT_END_OP_CONVERSION_CHECK(translate_map.count(decoder->get_op_type()),
            //                               "No translator found for " + decoder->get_op_type() + " node.");
            // auto op_fun = &(translate_map.at(decoder->get_op_type()));
            // ov::frontend::onnx::NodeContext node_context(decoder, inputs, submodel_translation_functions);
            const NodeProto* node_def = nullptr;
            decoder->experimental_get_internal_structures(reinterpret_cast<const void**>(&node_def));
            ov::frontend::onnx::Node node_context(*decoder, all_tensor_values);
            ov_outputs = (*translator)(node_context);
        } catch (...) {
            /*
            if (fail_fast) {
                if (m_telemetry && translator == nullptr) {
                    m_telemetry->send_event("error_cause", "tflite_" + decoder->get_op_type());
                }
                throw;
            } else {
                auto operation = std::make_shared<ov::frontend::onnx::FrameworkNode>(decoder, inputs, out_size);
                operation->set_friendly_name(decoder->get_op_name());
                ov_outputs = operation->outputs();
            }
            */
        }
        for (size_t i = 0; i < out_size; ++i) {
            const auto& name = decoder->get_output_tensor_name(i);
            all_tensor_values[name] = ov_outputs[i];
            all_tensor_places[name]->translate(all_tensor_values[name], !no_conversion);
        }
    }

    // outputs
    ResultVector results;
    results.reserve(model_onnx->get_outputs().size());
    for (const auto& output : model_onnx->get_outputs()) {
        const auto& tensor = std::dynamic_pointer_cast<ov::frontend::onnx::TensorONNXPlace>(output);
        FRONT_END_GENERAL_CHECK(tensor != nullptr,
                                "Inputs of ov::frontend::onnx::InputModel must be TensorLitePlace instances");
        const auto name = tensor->get_names()[0];
        if (!all_tensor_values.count(name)) {
            continue;
        }
        const auto& output_value = all_tensor_values[name];
        const auto& result = std::make_shared<ov::op::v0::Result>(output_value);
        auto input = result->output(0);
        tensor->translate(input, !no_conversion);
        result->set_friendly_name(name);
        results.push_back(result);
    }
    auto model_name = "onnx_Frontend_IR";
    ov_function = std::make_shared<ov::Model>(results, parameters, model_name);
#endif
}

std::shared_ptr<ov::Model> FrontEnd::decode_unify(const InputModel::Ptr& model) const {
    std::shared_ptr<ov::Model> ov_model;
    translate_graph(model, false, true, ov_model);
    return ov_model;
}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    if (auto telemetry = std::dynamic_pointer_cast<TelemetryExtension>(extension)) {
        m_extensions.telemetry = telemetry;
    } else if (auto transformation = std::dynamic_pointer_cast<DecoderTransformationExtension>(extension)) {
        m_transformation_extensions.push_back(transformation);
    } else if (const auto& so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(extension)) {
        add_extension(so_ext->extension());
        m_other_extensions.push_back(so_ext);
    } else if (auto common_conv_ext = ov::as_type_ptr<ov::frontend::ConversionExtension>(extension)) {
        m_extensions.conversions.push_back(common_conv_ext);
    } else if (const auto onnx_conv_ext = ov::as_type_ptr<onnx::ConversionExtension>(extension)) {
        m_extensions.conversions.push_back(onnx_conv_ext);
    } else if (auto progress_reporter = std::dynamic_pointer_cast<ProgressReporterExtension>(extension)) {
        m_extensions.progress_reporter = progress_reporter;
    } else if (auto op_base_ext = std::dynamic_pointer_cast<ov::BaseOpExtension>(extension)) {
        for (const auto& attached_ext : op_base_ext->get_attached_extensions()) {
            add_extension(attached_ext);
        }
    }
}
