// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/jax/frontend.hpp"

#include "input_model.hpp"
#include "jax_framework_node.hpp"
#include "op_table.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/frontend/jax/extension/conversion.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/util/log.hpp"
#include "translate_session.hpp"

namespace ov {
namespace frontend {
namespace jax {

namespace {
std::map<std::string, std::string> get_unconverted_types_from_model(const std::shared_ptr<Model>& model) {
    std::map<std::string, std::string> unconverted_ops_types;
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto& fw_node = ov::as_type_ptr<JaxFrameworkNode>(node)) {
            const auto& attrs = fw_node->get_attrs();
            FRONT_END_GENERAL_CHECK(attrs.find(JaxFrameworkNode::op_type_key) != attrs.end(),
                                    "FrameworkNode attributes do not contain operation type.");
            std::string exception_msg;
            if (attrs.find(JaxFrameworkNode::failed_conversion_key) != attrs.end()) {
                exception_msg = attrs.at(JaxFrameworkNode::failed_conversion_key);
            }
            if (!unconverted_ops_types.count(attrs.at(JaxFrameworkNode::op_type_key))) {
                unconverted_ops_types[attrs.at(JaxFrameworkNode::op_type_key)] = exception_msg;
            }
        }
        if (const auto& fw_node = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
            for (size_t i = 0; i < fw_node->get_internal_subgraphs_size(); i++) {
                const auto& internal_types = get_unconverted_types_from_model(fw_node->get_function(i));
                unconverted_ops_types.insert(internal_types.begin(), internal_types.end());
            }
        }
    }
    return unconverted_ops_types;
}

std::string pack_detailed_failure_report(const std::map<std::string, std::string>& unconverted_ops,
                                         const std::string& additional_error = "") {
    std::stringstream error_msg;
    std::stringstream unconverted_ops_msg;
    std::stringstream failed_ops_msg;
    std::stringstream failed_ops_short;
    error_msg << "Model wasn't fully converted.";
    unconverted_ops_msg << "-- No conversion rule found for operations: ";
    failed_ops_msg << " Failed operations detailed log:";
    failed_ops_short << "-- Conversion is failed for: ";
    bool at_least_one = false;
    bool at_least_one_except = false;
    for (auto&& op : unconverted_ops) {
        if (op.second.empty()) {
            if (at_least_one)
                unconverted_ops_msg << ", ";
            unconverted_ops_msg << op.first;
            at_least_one = true;
        } else {
            if (at_least_one_except)
                failed_ops_short << ", ";
            failed_ops_short << op.first;
            failed_ops_msg << "\n-- " << op.first << " with a message:\n" << op.second;
            at_least_one_except = true;
        }
    }
    if (at_least_one_except)
        error_msg << failed_ops_msg.str();
    error_msg << "\nSummary:" << additional_error;
    if (at_least_one)
        error_msg << '\n' << unconverted_ops_msg.str();
    if (at_least_one_except)
        error_msg << '\n' << failed_ops_short.str();
    return error_msg.str();
}
}  // namespace

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(const ov::frontend::InputModel::Ptr& model) const {
    FRONT_END_GENERAL_CHECK(std::dynamic_pointer_cast<jax::InputModel>(model), "Invalid input model");
    std::map<std::string, CreatorFunction> supported_ops = get_supported_ops(model);
    std::shared_ptr<Model> converted_model;
    {
        TranslateSession translate_session(model, supported_ops, m_telemetry);
        converted_model = translate_session.get_converted_model();
    }

    const auto& unconverted_ops = get_unconverted_types_from_model(converted_model);
    for (auto&& op : unconverted_ops) {
        if (m_telemetry) {
            m_telemetry->send_event("error_cause", "jax_" + op.first);
        }
    }
    bool is_conversion_successful = unconverted_ops.size() == 0;
    FRONT_END_OP_CONVERSION_CHECK(is_conversion_successful, pack_detailed_failure_report(unconverted_ops));
    return converted_model;
}

void FrontEnd::convert(const std::shared_ptr<Model>& partiallyConverted) const {
    FRONT_END_NOT_IMPLEMENTED(convert);
}

std::shared_ptr<Model> FrontEnd::convert_partially(const ov::frontend::InputModel::Ptr& model) const {
    FRONT_END_GENERAL_CHECK(std::dynamic_pointer_cast<jax::InputModel>(model), "Invalid input model");
    std::map<std::string, CreatorFunction> supported_ops = get_supported_ops(model);
    std::shared_ptr<Model> partial_model;
    {
        TranslateSession translate_session(model, supported_ops, m_telemetry);
        partial_model = translate_session.get_converted_model();
    }
    return partial_model;
}

std::shared_ptr<Model> FrontEnd::decode(const InputModel::Ptr& model) const {
    FRONT_END_NOT_IMPLEMENTED(decode);
}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    if (auto conv_ext = std::dynamic_pointer_cast<ov::frontend::ConversionExtension>(extension)) {
        m_conversion_extensions.push_back(conv_ext);
        m_op_extension_translators[conv_ext->get_op_type()] = [=](const NodeContext& context) {
            return conv_ext->get_converter()(context);
        };
    } else if (auto conv_ext = std::dynamic_pointer_cast<ov::frontend::jax::ConversionExtension>(extension)) {
        m_conversion_extensions.push_back(conv_ext);
        m_op_extension_translators[conv_ext->get_op_type()] = [=](const NodeContext& context) {
            return conv_ext->get_converter()(context);
        };
    } else if (const auto& so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(extension)) {
        add_extension(so_ext->extension());
        m_extensions.push_back(so_ext);
    } else if (const auto& telemetry = std::dynamic_pointer_cast<TelemetryExtension>(extension)) {
        m_telemetry = telemetry;
    } else if (auto op_base_ext = std::dynamic_pointer_cast<ov::BaseOpExtension>(extension)) {
        for (const auto& attached_ext : op_base_ext->get_attached_extensions()) {
            add_extension(attached_ext);
        }
    }
}

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    // Last boolean flag in `variants` (if presented) is reserved for FE
    // configuration
    size_t extra_variants_num = variants.size() > 0 && variants[variants.size() - 1].is<bool>() ? 1 : 0;
    // Currently Jax FrontEnd only support JaxDecoder as input
    if (variants.size() != 1 + extra_variants_num || !variants[0].is<std::shared_ptr<IDecoder>>())
        return false;
    auto decoder = variants[0].as<std::shared_ptr<IDecoder>>();
    return decoder && std::dynamic_pointer_cast<JaxDecoder>(decoder);
}

ov::frontend::InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    // Last boolean flag in `variants` (if presented) is reserved for FE
    // configuration
    size_t extra_variants_num = variants.size() > 0 && variants[variants.size() - 1].is<bool>() ? 1 : 0;
    FRONT_END_GENERAL_CHECK(variants.size() == 1 + extra_variants_num,
                            "Jax Frontend supports exactly one parameter in "
                            "model representation, got ",
                            std::to_string(variants.size()),
                            " instead.");
    FRONT_END_GENERAL_CHECK(variants[0].is<std::shared_ptr<IDecoder>>(),
                            "Jax Frontend doesn't support provided model type. "
                            "Please provide supported model "
                            "object using Python API.");
    auto decoder = variants[0].as<std::shared_ptr<IDecoder>>();
    auto tdecoder = std::dynamic_pointer_cast<JaxDecoder>(decoder);
    FRONT_END_GENERAL_CHECK(tdecoder, "Couldn't cast ov::Any to JaxDecoder");
    return std::make_shared<jax::InputModel>(tdecoder);
}

std::map<std::string, CreatorFunction> FrontEnd::get_supported_ops(const ov::frontend::InputModel::Ptr& model) const {
    std::map<std::string, CreatorFunction> supported_ops;
    supported_ops = get_supported_ops_jaxpr();
    for (auto i = m_op_extension_translators.begin(); i != m_op_extension_translators.end(); i++)
        supported_ops[i->first] = i->second;
    return supported_ops;
}

}  // namespace jax
}  // namespace frontend
}  // namespace ov
