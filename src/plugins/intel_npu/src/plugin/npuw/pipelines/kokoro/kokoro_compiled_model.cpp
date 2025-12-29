// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kokoro_compiled_model.hpp"

#include "kokoro_infer_request.hpp"
#include "npuw/logging.hpp"
#include "kokoro_split.hpp"
#include "intel_npu/config/config.hpp"
#include "plugin.hpp"

namespace {
    // Remove all options "NPUW_.*"
    ov::AnyMap remove_all_npuw_parameters(const ov::AnyMap& properties) {
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
}

bool ov::npuw::KokoroCompiledModel::is_kokoro_monolithic_model(const std::shared_ptr<ov::Model>& model) {
    if (!model) {
        return false;
    }

    bool has_en = false;
    bool has_asr = false;
    bool has_repeat_interleve_range = false;

    for (const auto& op : model->get_ops()) {
        const auto& name = op->get_friendly_name();
        if (name == "aten::repeat_interleave/Range") {
            has_repeat_interleve_range = true;
        } else if (name == "aten::matmul/MatMul") {
            has_en = true;
        } else if (name == "aten::matmul/MatMul_1") {
            has_asr = true;
        }
        if (has_en && has_asr && has_repeat_interleve_range) {
            return true;
        }
    }
    return false;
}

ov::npuw::KokoroCompiledModel::KokoroCompiledModel(const std::shared_ptr<ov::Model>& model,
                                                   const std::shared_ptr<const ov::IPlugin>& plugin,
                                                   const ov::AnyMap& properties)
    : ov::npuw::ICompiledModel(model, plugin),
      m_name(model->get_friendly_name()) {
    LOG_DEBUG("Creating KokoroCompiledModel");

    // Decompose kokoro model into two static models 
    KokoroSplitResult split_result = KokoroSplit::split_model(model, m_kokoro_cfg);

    LOG_DEBUG("Compiling kokoro model A.");
    // Model A doesn't require decomposition, so it should be handled by CPU or NPU plugin 
    if (is_cpu_only(properties)) {
        auto core = plugin->get_core();
        m_model_a_compiled = core->compile_model(split_result.model_a, "CPU", ov::AnyMap{});
    } else {
        // Plugin don't have to know about NPUW parameters 
        ov::AnyMap model_a_properties = remove_all_npuw_parameters(properties);
        m_model_a_compiled = plugin->compile_model(split_result.model_a, model_a_properties);
    }
    
    LOG_DEBUG("Compiling kokoro model B.");
    ov::AnyMap properties_model_b = properties;

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
    // FIXME Not implemented
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_NOT_IMPLEMENTED;
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