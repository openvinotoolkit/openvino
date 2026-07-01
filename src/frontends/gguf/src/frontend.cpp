// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/frontend.h"

#include "input_model.h"
#include "op_table.h"
#include "openvino/core/so_extension.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/gguf/decoder.h"
#include "openvino/frontend/manager.hpp"
#include "translate_session.h"

namespace ov {
namespace frontend {
namespace gguf {

// NOTE ON DISCOVERABILITY (intentional):
// This frontend is consumed only by direct linkage -- a caller (the llama.cpp ggml-openvino
// backend, OpenVINO GenAI) links openvino::frontend::gguf, constructs ov::frontend::gguf::FrontEnd
// and feeds it a GgufDecoder. It has no file-based reader (see supported_impl / load_impl: it only
// accepts a live GgufDecoder, never a .gguf path).
//
// The library is installed alongside the other frontends, so it exports the standard plugin
// registration entry points (get_api_version / get_front_end_data) -- otherwise FrontEndManager
// would throw while scanning the frontend directory. It registers itself as HIDDEN
// (FrontEndPluginInfo::m_hidden), which keeps it out of get_available_front_ends() and prevents
// load_by_model / load_by_framework from ever selecting it. supported_impl() also returns false.
// Direct linkers are unaffected. Flip m_hidden to false only once this frontend gains file-based
// loading and passes production review.

struct FrontEnd::Impl {
    std::unordered_map<std::string, CreatorFunction> op_extension_translators;
    std::vector<ConversionExtensionBase::Ptr> conversion_extensions;
    // Transformation extensions run in the normalization stage. A caller uses these to swap the
    // default (stateless) SetRows lowering for an alternative (e.g. a backend stateful lowering).
    std::vector<DecoderTransformationExtension::Ptr> transformation_extensions;
    TelemetryExtension::Ptr telemetry;
};

namespace {

// Merge the built-in op translators with any registered via ConversionExtension
// (extension translators win on name collision).
std::unordered_map<std::string, CreatorFunction> merged_ops(
    const std::unordered_map<std::string, CreatorFunction>& ext_translators) {
    auto ops = get_supported_ops();
    for (const auto& ext : ext_translators) {
        ops[ext.first] = ext.second;
    }
    return ops;
}

}  // namespace

FrontEnd::FrontEnd() : m_impl(std::make_shared<Impl>()) {}
FrontEnd::~FrontEnd() = default;

std::shared_ptr<Model> FrontEnd::convert(const InputModel::Ptr& model) const {
    auto gguf_model = std::dynamic_pointer_cast<gguf::InputModel>(model);
    FRONT_END_GENERAL_CHECK(gguf_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    {
        auto ops = merged_ops(m_impl->op_extension_translators);
        TranslateSession translate_session(model, ops, gguf_model->is_naive(), m_impl->transformation_extensions);
        converted_model = translate_session.get_converted_model();
    }
    return converted_model;
}

std::string FrontEnd::get_name() const {
    return "gguf";
}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    if (auto conv_ext = ov::as_type_ptr<ov::frontend::ConversionExtension>(extension)) {
        m_impl->conversion_extensions.push_back(conv_ext);
        // Wrap the base CreatorFunction (takes ov::frontend::NodeContext) into a
        // gguf::CreatorFunction (takes gguf::NodeContext). gguf::NodeContext IS an
        // ov::frontend::NodeContext so the call is safe.
        m_impl->op_extension_translators[conv_ext->get_op_type()] = [conv_ext](const NodeContext& ctx) {
            return conv_ext->get_converter()(ctx);
        };
    } else if (const auto& so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(extension)) {
        add_extension(so_ext->extension());
        m_extensions.push_back(so_ext);
    } else if (const auto& transformation = std::dynamic_pointer_cast<DecoderTransformationExtension>(extension)) {
        m_impl->transformation_extensions.push_back(transformation);
    } else if (const auto& telemetry = std::dynamic_pointer_cast<TelemetryExtension>(extension)) {
        m_impl->telemetry = telemetry;
    } else if (auto op_base_ext = std::dynamic_pointer_cast<ov::BaseOpExtension>(extension)) {
        for (const auto& attached_ext : op_base_ext->get_attached_extensions()) {
            add_extension(attached_ext);
        }
    }
}

bool FrontEnd::supported_impl(const std::vector<ov::Any>&) const {
    // Always false: this frontend is never selected by FrontEndManager (load_by_model). It is used
    // only via direct linkage -- a caller constructs FrontEnd and calls convert() with an
    // InputModel built from a GgufDecoder -- which does not go through supported(). See the
    // discoverability note at the top of this file.
    return false;
}

InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    FRONT_END_GENERAL_CHECK(!variants.empty(),
                            "GGUF Frontend requires at least one parameter in model representation.");
    FRONT_END_GENERAL_CHECK(variants[0].is<std::shared_ptr<GgufDecoder>>(),
                            "GGUF Frontend supports loading from a GgufDecoder only.");
    auto decoder = variants[0].as<std::shared_ptr<GgufDecoder>>();
    FRONT_END_GENERAL_CHECK(decoder, "Couldn't cast ov::Any to std::shared_ptr<GgufDecoder>");
    return std::make_shared<InputModel>(decoder);
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov

// Plugin registration. The frontend is installed in the frontend directory, so it must export
// these entry points or FrontEndManager throws while scanning it. It registers as hidden (see the
// discoverability note above): FrontEndManager loads it without error but never lists or
// auto-selects it; only direct linkers use it.
GGUF_FRONTEND_C_API ov::frontend::FrontEndVersion get_api_version() {
    return OV_FRONTEND_API_VERSION;
}

GGUF_FRONTEND_C_API void* get_front_end_data() {
    auto* res = new ov::frontend::FrontEndPluginInfo();
    res->m_name = "gguf";
    res->m_creator = []() {
        return std::make_shared<ov::frontend::gguf::FrontEnd>();
    };
    res->m_hidden = true;
    return res;
}
