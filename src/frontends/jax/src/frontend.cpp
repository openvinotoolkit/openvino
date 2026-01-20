// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/jax/frontend.hpp"

#include <optional>

#include "input_model.hpp"
#include "jax_framework_node.hpp"
#include "op_table.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/frontend/jax/extension/conversion.hpp"
#include "openvino/frontend/unconverted_ops_report.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/util/log.hpp"
#include "translate_session.hpp"

namespace ov {
namespace frontend {
namespace jax {

namespace {

const std::vector<ov::frontend::UnconvertedOpExtractor>& get_unconverted_extractors() {
    static const std::vector<ov::frontend::UnconvertedOpExtractor> extractors{
        ov::frontend::make_unconverted_op_extractor<JaxFrameworkNode>(
            [](const std::shared_ptr<JaxFrameworkNode>& node) -> std::optional<ov::frontend::FrameworkNodeErrorInfo> {
                const auto& attrs = node->get_attrs();
                FRONT_END_GENERAL_CHECK(attrs.find(JaxFrameworkNode::op_type_key) != attrs.end(),
                                        "FrameworkNode attributes do not contain operation type.");
                return ov::frontend::build_framework_node_error_info(attrs,
                                                                     JaxFrameworkNode::op_type_key,
                                                                     JaxFrameworkNode::failed_conversion_key);
            })};
    return extractors;
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

    const auto unconverted_ops = ov::frontend::collect_unconverted_ops(converted_model, get_unconverted_extractors());
    if (m_telemetry) {
        for (const auto& entry : unconverted_ops) {
            m_telemetry->send_event("error_cause", "jax_" + entry.first);
        }
    }
    FRONT_END_OP_CONVERSION_CHECK(
        unconverted_ops.empty(),
        ov::frontend::format_unconverted_ops_report(unconverted_ops,
                                                    std::string{},
                                                    "[JAX Frontend] Model wasn't fully converted."));
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
    if (auto conv_ext = ov::as_type_ptr<ov::frontend::ConversionExtension>(extension)) {
        m_conversion_extensions.push_back(conv_ext);
        m_op_extension_translators[conv_ext->get_op_type()] = [=](const NodeContext& context) {
            return conv_ext->get_converter()(context);
        };
    } else if (auto conv_ext = ov::as_type_ptr<ov::frontend::jax::ConversionExtension>(extension)) {
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
