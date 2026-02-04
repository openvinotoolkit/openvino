// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kokoro_compiled_model.hpp"

#include "intel_npu/config/config.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "kokoro_infer_request.hpp"
#include "kokoro_split.hpp"
#include "npuw/logging.hpp"
#include "plugin.hpp"

namespace {
// Remove all options "NPUW_.*"
ov::AnyMap without_npuw_params(const ov::AnyMap& properties) {
    ov::AnyMap result;
    for (const auto& item : properties) {
        if (item.first.find("NPUW") == std::string::npos) {
            result.insert(item);
        }
    }
    return result;
}

// Check properties for NPUW_DEVICES options and return true if only CPU is present
bool is_cpu_only(const ov::AnyMap& properties) {
    auto it = properties.find("NPUW_DEVICES");
    if (it != properties.end()) {
        return it->second.as<std::string>() == "CPU";
    }
    return false;
}

void split_kokoro_properties(const ov::AnyMap& properties,
                             ov::AnyMap& other_properties,
                             ov::AnyMap& kokoro_properties) {
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        if (it->first.find("NPUW_KOKORO") != it->first.npos) {
            kokoro_properties.insert(*it);
        } else {
            other_properties.insert(*it);
        }
    }
}

std::map<std::string, std::string> any_copy(const ov::AnyMap& params) {
    std::map<std::string, std::string> result;
    for (auto&& value : params) {
        if (value.second.is<std::string>()) {
            result.emplace(value.first, value.second.as<std::string>());
        } else if (value.second.is<bool>()) {
            result.emplace(value.first, value.second.as<bool>() ? "YES" : "NO");
        } else {
            std::stringstream ss;
            value.second.print(ss);
            result.emplace(value.first, ss.str());
        }
    }
    return result;
}
}  // namespace

ov::npuw::KokoroCompiledModel::KokoroCompiledModel(const std::shared_ptr<ov::Model>& model,
                                                   const std::shared_ptr<const ov::IPlugin>& plugin,
                                                   const ov::AnyMap& properties)
    : ov::npuw::ICompiledModel(model, plugin),
      m_name(model->get_friendly_name()),
      m_options_desc(std::make_shared<::intel_npu::OptionsDesc>()),
      m_cfg(m_options_desc) {
    LOG_DEBUG("Creating KokoroCompiledModel");

    ::intel_npu::registerNPUWKokoroOptions(*m_options_desc);

    // Split properties to separate Kokoro specific options and NPU plugin options
    ov::AnyMap npuw_kokoro_props;
    ov::AnyMap common_props;

    split_kokoro_properties(properties, common_props, npuw_kokoro_props);

    m_cfg.parseEnvVars();
    m_cfg.update(any_copy(npuw_kokoro_props));

    // Get configuration from m_cfg, which now has defaults, env vars and user properties merged
    m_kokoro_cfg.block_size = m_cfg.get<::intel_npu::NPUW_KOKORO_BLOCK_SIZE>();
    m_kokoro_cfg.overlap_size = m_cfg.get<::intel_npu::NPUW_KOKORO_OVERLAP_SIZE>();

    // Decompose kokoro model into two static models
    KokoroSplitResult split_result = KokoroSplit::split_model(model, m_kokoro_cfg);

    LOG_DEBUG("Compiling kokoro model A...");
    // Model A doesn't require decomposition, so it should be handled by CPU or NPU plugin
    if (is_cpu_only(common_props)) {
        auto core = plugin->get_core();
        m_model_a_compiled = core->compile_model(split_result.model_a, "CPU", ov::AnyMap{});
    } else {
        // Plugin don't have to know about NPUW parameters
        ov::AnyMap model_a_properties = without_npuw_params(common_props);
        m_model_a_compiled = plugin->compile_model(split_result.model_a, model_a_properties);
    }

    LOG_DEBUG("Compiling kokoro model B...");
    ov::AnyMap properties_model_b = common_props;

    // Enforce offloading to CPU for non-accurate subgraphs
    if (!properties_model_b.count("NPUW_ONLINE_PIPELINE")) {
        // REP mode is giving best compile time / stability results
        properties_model_b["NPUW_ONLINE_PIPELINE"] = "REP";
    }
    if (!properties_model_b.count("NPUW_SUBMODEL_DEVICE")) {
        // 20 & 99 - ISTFT & STFT, 55 & 62 - Not accurate
        properties_model_b["NPUW_SUBMODEL_DEVICE"] = "20:CPU,99:CPU,55:CPU,62:CPU";
    }
    m_model_b_compiled = std::dynamic_pointer_cast<ov::npuw::ICompiledModel>(
        ov::npuw::ICompiledModel::create(split_result.model_b, plugin, properties_model_b));
}

void ov::npuw::KokoroCompiledModel::export_model(std::ostream& stream) const {
    // FIXME Not implemented
    LOG_DEBUG("Exporting KokoroCompiledModel");
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::npuw::KokoroCompiledModel> ov::npuw::KokoroCompiledModel::import_model(
    std::istream& stream,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    // FIXME Not implemented
    LOG_DEBUG("Importing KokoroCompiledModel");
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const ov::Model> ov::npuw::KokoroCompiledModel::get_runtime_model() const {
    // FIXME Not implemented
    LOG_DEBUG("Getting runtime model from KokoroCompiledModel");
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::npuw::KokoroCompiledModel::set_property(const ov::AnyMap& properties) {
    // FIXME Not implemented
    LOG_DEBUG("Setting properties to KokoroCompiledModel");
    OPENVINO_NOT_IMPLEMENTED;
}

ov::Any ov::npuw::KokoroCompiledModel::get_property(const std::string& name) const {
    OPENVINO_SUPPRESS_DEPRECATED_START
    if (m_model_b_compiled) {
        return m_model_b_compiled->get_property(name);
    }
    OPENVINO_THROW("Property ", name, " not found");
    OPENVINO_SUPPRESS_DEPRECATED_END
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::KokoroCompiledModel::create_sync_infer_request() const {
    auto* non_const_this = const_cast<ov::npuw::KokoroCompiledModel*>(this);
    return non_const_this->create_kokoro_infer_request();
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::KokoroCompiledModel::create_kokoro_infer_request() {
    auto this_sptr = std::static_pointer_cast<ov::npuw::KokoroCompiledModel>(shared_from_this());
    return std::make_shared<ov::npuw::KokoroInferRequest>(this_sptr);
}
