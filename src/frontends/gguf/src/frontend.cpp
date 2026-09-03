// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/frontend.hpp"

#include <filesystem>
#include <fstream>

#include "builder/arch_registry.hpp"
#include "builder/gguf_builder.hpp"
#include "builder/gguf_builder_decoder.hpp"
#include "input_model.hpp"
#include "op_table.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/frontend/common/path_util.hpp"
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/gguf/decoder.hpp"
#include "openvino/frontend/gguf/extension/architecture.hpp"
#include "openvino/frontend/manager.hpp"
#include "translate_session.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// This frontend has two ingest paths, both converging on the same GgufDecoder + op translators:
//   1. a live GgufDecoder passed in by a direct linker (the llama.cpp ggml-openvino cgraph path);
//   2. a .gguf file path (the OpenVINO-native path): the frontend parses the container and builds
//      the transformer graph per-architecture via the native builder (see load_impl Path 2).
//
// Discoverability: "gguf" is in manager.cpp's is_hidden_frontend list, so it is not advertised by
// available_front_ends() and not auto-selected by load_by_model (core.read_model(".gguf") does
// not resolve to it). It is still reachable explicitly, by direct linkage or by name via
// load_by_framework("gguf"). supported_impl below stays implemented, so enabling core.read_model
// later is just dropping the name from that list.
//
// Driving the frontend directly needs no follow-up pass: normalization runs inside convert(), and
// the only step read_model adds, update_v10_model(), fires solely for legacy IR v10.

struct FrontEnd::Impl {
    std::unordered_map<std::string, CreatorFunction> op_extension_translators;
    std::vector<ConversionExtensionBase::Ptr> conversion_extensions;
    // Transformation extensions run in the normalization stage. A caller uses these to swap the
    // default (stateless) SetRows lowering for an alternative (e.g. a backend stateful lowering).
    std::vector<DecoderTransformationExtension::Ptr> transformation_extensions;
    TelemetryExtension::Ptr telemetry;
    // Architectures this frontend accepts: the built-in set plus any registered at runtime.
    // Per-instance, like the extension lists above, so two frontends do not share registrations.
    ArchRegistry arch_registry;
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

// True if the file at `path` begins with the GGUF magic ("GGUF").
bool has_gguf_magic(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    char magic[4] = {};
    f.read(magic, sizeof(magic));
    return f.gcount() == 4 && magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' && magic[3] == 'F';
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
        TranslateSession translate_session(model, ops, m_impl->transformation_extensions);
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
    } else if (const auto& arch_ext = std::dynamic_pointer_cast<ArchitectureExtension>(extension)) {
        m_impl->arch_registry.add_extension(arch_ext);
    } else if (const auto& telemetry = std::dynamic_pointer_cast<TelemetryExtension>(extension)) {
        m_impl->telemetry = telemetry;
    } else if (auto op_base_ext = std::dynamic_pointer_cast<ov::BaseOpExtension>(extension)) {
        for (const auto& attached_ext : op_base_ext->get_attached_extensions()) {
            add_extension(attached_ext);
        }
    }
}

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    // Two accepted inputs:
    //  1. a GgufDecoder (the llama.cpp cgraph path passes one in directly), or
    //  2. a path to a .gguf file (the OpenVINO-native path; sniff the GGUF magic).
    if (variants.empty()) {
        return false;
    }
    if (variants[0].is<std::shared_ptr<GgufDecoder>>()) {
        return true;
    }
    if (auto path = ov::frontend::get_path_from_any(variants[0])) {
        std::filesystem::path model_path = std::move(*path);
        return model_path.extension() == ".gguf" && has_gguf_magic(model_path);
    }
    return false;
}

InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    FRONT_END_GENERAL_CHECK(!variants.empty(),
                            "GGUF Frontend requires at least one parameter in model representation.");

    // Path 1: a GgufDecoder passed in directly (e.g. llama.cpp cgraph decoder).
    if (variants[0].is<std::shared_ptr<GgufDecoder>>()) {
        auto decoder = variants[0].as<std::shared_ptr<GgufDecoder>>();
        FRONT_END_GENERAL_CHECK(decoder, "Couldn't cast ov::Any to std::shared_ptr<GgufDecoder>");
        return std::make_shared<InputModel>(decoder);
    }

    // Path 2: a .gguf file path -> native builder -> GgufBuilderDecoder.
    if (auto path = ov::frontend::get_path_from_any(variants[0])) {
        std::filesystem::path model_path = std::move(*path);
        FRONT_END_GENERAL_CHECK(model_path.extension() == ".gguf",
                                "GGUF Frontend file loading expects a .gguf file, got: ",
                                model_path.string());
        auto graph = build_ggml_graph_from_gguf(model_path.string(), m_impl->arch_registry);
        auto decoder = std::make_shared<GgufBuilderDecoder>(graph);
        return std::make_shared<InputModel>(decoder);
    }

    FRONT_END_GENERAL_CHECK(false,
                            "GGUF Frontend doesn't support the provided model representation. Provide a GgufDecoder "
                            "or a path to a .gguf file.");
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov

// Plugin registration. Exports the standard entry points so FrontEndManager can load the library;
// selection is covered by the discoverability note at the top of this file.
GGUF_FRONTEND_C_API ov::frontend::FrontEndVersion get_api_version() {
    return OV_FRONTEND_API_VERSION;
}

GGUF_FRONTEND_C_API void* get_front_end_data() {
    auto* res = new ov::frontend::FrontEndPluginInfo();
    res->m_name = "gguf";
    res->m_creator = []() {
        return std::make_shared<ov::frontend::gguf::FrontEnd>();
    };
    return res;
}
