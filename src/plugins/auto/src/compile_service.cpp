// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compile_service.hpp"

#include <chrono>

#include "cache_manager.hpp"

namespace ov {
namespace auto_plugin {

CompileService::CompileService(std::shared_ptr<ov::ICore> core, std::filesystem::path model_path, std::string log_tag)
    : m_core(std::move(core)),
      m_model_path(std::move(model_path)),
      m_log_tag(std::move(log_tag)) {}

CompileService::~CompileService() {
    stop_precompile();
}

SoCompiledModel CompileService::compile(const std::shared_ptr<const ov::Model>& model,
                                        const DeviceName& device,
                                        const ov::AnyMap& config) {
    if (m_precompiler) {
        m_precompiler->claim_device(device);
    }
    const auto start = std::chrono::steady_clock::now();
    auto compiled_model = m_model_path.empty() ? m_core->compile_model(model, device, config)
                                               : m_core->compile_model(m_model_path, device, config);
    const auto compile_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::optional<bool> loaded_from_cache;
    try {
        loaded_from_cache = compiled_model->get_property(ov::loaded_from_cache.name()).as<bool>();
    } catch (const ov::Exception&) {
    }
    std::optional<std::string> blob_id;
    try {
        if (CacheManager::is_reloadable(*m_core, device, config)) {
            blob_id = CacheManager::get_blob_id(*m_core, device, config, m_model_path.empty() ? model : nullptr, m_model_path);
        }
    } catch (const std::exception& ex) {
        LOG_DEBUG_TAG("Device: [%s]: cache blob ID is unknown: %s", device.c_str(), ex.what());
    }
    LOG_INFO_TAG("Device: [%s]: Compile model took %lf ms, loaded from cache: %s",
                 device.c_str(),
                 compile_ms,
                 loaded_from_cache ? (*loaded_from_cache ? "yes" : "no") : "unknown");
    std::lock_guard<std::mutex> lock(m_stats_mutex);
    auto& stats = m_stats[device];
    stats.last_compile_ms = compile_ms;
    stats.last_loaded_from_cache = loaded_from_cache;
    stats.last_blob_id = std::move(blob_id);
    stats.compile_count++;
    return compiled_model;
}

std::optional<CompileStats> CompileService::stats(const DeviceName& device) const {
    std::lock_guard<std::mutex> lock(m_stats_mutex);
    const auto iter = m_stats.find(device);
    return iter == m_stats.end() ? std::nullopt : std::optional<CompileStats>(iter->second);
}

void CompileService::enable_precompile(CacheManager& manager, std::shared_ptr<const ov::Model> model) {
    OPENVINO_ASSERT(!m_precompiler, "The background compilation lane is already enabled");
    m_precompiler = std::make_shared<CachePrecompiler>(manager, m_core, std::move(model), m_model_path, m_log_tag);
}

bool CompileService::has_precompile() const {
    return m_precompiler != nullptr;
}

void CompileService::start_precompile(const DeviceName& actual_device,
                                      const std::vector<DeviceInformation>& candidates,
                                      CachePrecompiler::DeviceLoadedCheck is_device_loaded) {
    if (m_precompiler) {
        m_precompiler->start(actual_device, candidates, std::move(is_device_loaded));
    }
}

void CompileService::stop_precompile() {
    if (m_precompiler) {
        m_precompiler->stop();
    }
}

const std::string& CompileService::get_log_tag() const noexcept {
    return m_log_tag;
}

}  // namespace auto_plugin
}  // namespace ov
