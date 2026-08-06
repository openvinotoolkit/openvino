// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_compiled_model.hpp"

#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include "intel_npu/config/npuw.hpp"
#include "logging.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/properties.hpp"
#include "util.hpp"

namespace {

bool is_kv_cache_name(const std::string& name) {
    return ov::npuw::util::starts_with(name, "key_cache.") || ov::npuw::util::starts_with(name, "value_cache.");
}

}  // anonymous namespace

ov::npuw::PACompiledModel::PreparedState ov::npuw::PACompiledModel::prepare(
    const std::shared_ptr<ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    // The fallback device is an internal development knob, not a config
    // option - an env var keeps it out of user configs (and blob cache keys).
    const char* device_env = std::getenv("OPENVINO_NPUW_PA_DEVICE");
    std::string device = (device_env != nullptr && device_env[0] != '\0') ? device_env : "CPU";
    // NPU cannot take the PA op itself (dynamic shapes, no PA kernel).
    OPENVINO_ASSERT(!ov::npuw::util::starts_with(device, "NPU"),
                    "OPENVINO_NPUW_PA_DEVICE must be the PagedAttention fallback device (CPU or GPU), got ",
                    device);

    LOG_INFO("PA: compiling the dynamic PA model 1:1 on " << device);

    // Sanity: this must be the model the CB pipeline deploys -- PA control
    // inputs plus a paged KV cache.
    bool has_past_lens = false, has_cache = false;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        has_past_lens |= (name == "past_lens");
        has_cache |= is_kv_cache_name(name);
    }
    OPENVINO_ASSERT(has_past_lens && has_cache,
                    "PACompiledModel expects the continuous-batching PA model "
                    "(past_lens + key_cache/value_cache inputs)");

    // The 1:1 part: the model is compiled exactly as received. NPUW_*,
    // NPU_USE_NPUW and NPU_* keys are this plugin's configuration and must not
    // reach the executing device (which would reject them as unsupported);
    // everything else (e.g. KV_CACHE_PRECISION, performance hints) is the
    // executing device's business and is forwarded.
    ov::AnyMap inner_config;
    for (const auto& [key, value] : properties) {
        if (ov::npuw::util::starts_with(key, "NPU")) {
            continue;
        }
        inner_config.emplace(key, value);
    }
    auto compiled = plugin->get_core()->compile_model(model, device, inner_config);
    OPENVINO_ASSERT(compiled != nullptr, "PACompiledModel requires a valid inner compiled model");

    // Stamp the device-resolved KV cache element types and shapes back onto
    // the model's cache Parameters. The source PA model declares them fully
    // dynamic (even the element type); the CB pipeline's KVCacheManager reads
    // cache precision and block geometry from *this* compiled model's ports,
    // so they must expose what the device actually decided.
    std::unordered_map<std::string, ov::Output<const ov::Node>> inner_inputs;
    for (const auto& input : compiled->inputs()) {
        inner_inputs.emplace(input.get_any_name(), input);
    }
    for (const auto& param : model->get_parameters()) {
        const auto& name = param->get_output_tensor(0).get_any_name();
        if (!is_kv_cache_name(name)) {
            continue;
        }
        auto it = inner_inputs.find(name);
        OPENVINO_ASSERT(it != inner_inputs.end(), "PA: inner compiled model lost the '", name, "' input");
        param->set_element_type(it->second.get_element_type());
        param->set_partial_shape(it->second.get_partial_shape());
    }
    model->validate_nodes_and_infer_types();

    return PreparedState{model, std::move(compiled), std::move(device)};
}

ov::npuw::PACompiledModel::PACompiledModel(const std::shared_ptr<ov::Model>& model,
                                           const std::shared_ptr<const ov::IPlugin>& plugin,
                                           const ov::AnyMap& properties)
    : PACompiledModel(prepare(model, plugin, properties), plugin) {}

ov::npuw::PACompiledModel::PACompiledModel(PreparedState prepared, const std::shared_ptr<const ov::IPlugin>& plugin)
    : ov::npuw::ICompiledModel(prepared.model, plugin),
      m_device(std::move(prepared.device)),
      m_compiled_model(std::move(prepared.compiled)) {
    // The device fixes the KV cache geometry at compile time; remember the
    // block size for validating block-table coverage per dispatch.
    for (const auto& input : m_compiled_model->inputs()) {
        if (input.get_any_name() == "key_cache.0") {
            const auto& shape = input.get_partial_shape();
            // [num_blocks (dyn), kv_heads, block_size, head_size]
            if (shape.rank().is_static() && shape.rank().get_length() == 4 && shape[2].is_static()) {
                m_block_size = static_cast<std::size_t>(shape[2].get_length());
            }
            break;
        }
    }
    LOG_INFO("PA: KV block_size fixed by " << m_device << ": " << m_block_size);
}

void ov::npuw::PACompiledModel::export_model(std::ostream&) const {
    OPENVINO_THROW_NOT_IMPLEMENTED("PACompiledModel does not support export_model()");
}

std::shared_ptr<const ov::Model> ov::npuw::PACompiledModel::get_runtime_model() const {
    return m_compiled_model->get_runtime_model();
}

void ov::npuw::PACompiledModel::set_property(const ov::AnyMap& properties) {
    // The PA-level options are fixed at compile time; catching them here gives
    // a clear error instead of the executing device's "unsupported property".
    for (const auto& [key, value] : properties) {
        if (ov::npuw::util::starts_with(key, "NPU")) {
            OPENVINO_THROW("PACompiledModel: '", key, "' cannot be changed after the model is compiled");
        }
    }
    m_compiled_model->set_property(properties);
}

ov::Any ov::npuw::PACompiledModel::get_property(const std::string& name) const {
    // The PA-level key is answered here; everything else is the executing
    // device's business (notably ov::execution_devices, which the CB pipeline
    // queries to pick its block size).
    if (name == std::string(::intel_npu::NPUW_PA::key())) {
        return true;
    }
    if (name == ov::supported_properties.name()) {
        // Keep the property surface self-consistent: the inner device's list
        // plus the PA key answered above.
        auto props = m_compiled_model->get_property(name).as<std::vector<ov::PropertyName>>();
        props.emplace_back(std::string(::intel_npu::NPUW_PA::key()), ov::PropertyMutability::RO);
        return props;
    }
    return m_compiled_model->get_property(name);
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::PACompiledModel::create_sync_infer_request() const {
    // The PA dispatcher (per-dispatch contract validation and execution)
    // arrives with the next change in this series.
    OPENVINO_THROW_NOT_IMPLEMENTED("PACompiledModel does not create infer requests yet");
}
