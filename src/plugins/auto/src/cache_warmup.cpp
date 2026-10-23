// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cache_warmup.hpp"

#include <algorithm>
#include <chrono>
#include <optional>

#include "openvino/runtime/compilation_context.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/util/file_util.hpp"

namespace ov {
namespace auto_plugin {
namespace {
// Lowers the host-side cost of the background compilation without changing the cache key.
ov::AnyMap make_warmup_config(const ov::ICore& core, const DeviceInformation& device) {
    std::vector<ov::PropertyName> supported;
    try {
        supported = core.get_property(device.device_name, ov::supported_properties);
    } catch (const ov::Exception&) {
        return device.config;
    }
    const auto is_supported = [&supported](const std::string& name) {
        return std::find(supported.begin(), supported.end(), name) != supported.end();
    };
    auto config = device.config;
    if (is_supported(ov::intel_gpu::hint::host_task_priority.name())) {
        config.emplace(ov::intel_gpu::hint::host_task_priority(ov::hint::Priority::LOW));
    }
    if (is_supported(ov::compilation_num_threads.name())) {
        config.emplace(ov::compilation_num_threads(std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 4)));
    }
    if (config.size() == device.config.size()) {
        return config;
    }
    const auto base_name = ov::DeviceIDParser(device.device_name).get_device_name();
    if (core.create_compile_config(base_name, config) != core.create_compile_config(base_name, device.config)) {
        return device.config;
    }
    return config;
}
}  // namespace

// The caller may modify its model once compile_model() returns, so warm-up works on a private clone; a clone failure
// propagates to the caller.
CacheWarmup::CacheWarmup(CacheWarmupManager& manager,
                         std::weak_ptr<ov::ICore> core,
                         std::shared_ptr<const ov::Model> model,
                         std::filesystem::path model_path,
                         std::string log_tag)
    : m_manager(manager),
      m_core(std::move(core)),
      m_model(model ? model->clone() : nullptr),
      m_model_path(std::move(model_path)),
      m_log_tag(std::move(log_tag)) {}

std::string CacheWarmup::get_log_tag() const noexcept {
    return m_log_tag;
}

void CacheWarmup::start(const std::string& actual_device, const std::vector<DeviceInformation>& candidates) {
    std::vector<DeviceInformation> devices;
    for (const auto& device : candidates) {
        if (device.device_name != actual_device && ov::DeviceIDParser(device.device_name).get_device_name() != "CPU") {
            devices.push_back(device);
        }
    }
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_started || m_stopped) {
            return;
        }
        m_started = true;
        if (devices.empty() || (!m_model && m_model_path.empty())) {
            m_model.reset();
            return;
        }
        m_devices = std::move(devices);
        LOG_INFO_TAG("start cache pre-compilation for %lu device(s)", static_cast<unsigned long>(m_devices.size()));
    }
    m_manager.enqueue(shared_from_this());
}

void CacheWarmup::stop() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_stopped = true;
    m_model.reset();
    m_cv.wait(lock, [this] {
        return !m_compiling;
    });
}

void CacheWarmup::run() {
    std::shared_ptr<const ov::Model> model;
    std::vector<DeviceInformation> devices;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        model = std::move(m_model);
        devices = std::move(m_devices);
    }
    for (const auto& device : devices) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_stopped) {
                return;
            }
            m_compiling = true;
        }
        const auto begin = std::chrono::steady_clock::now();
        try {
            if (auto core = m_core.lock()) {
                const auto config = make_warmup_config(*core, device);
                const auto cache_location =
                    CacheWarmupManager::get_cache_location(*core, config, device.device_name);
                if (cache_location.empty()) {
                    LOG_INFO_TAG("cache pre-compilation skipped for device: %s, caching is disabled for it",
                                 device.device_name.c_str());
                } else if (CacheWarmupManager::is_blob_cached(*core,
                                                              device.device_name,
                                                              config,
                                                              model,
                                                              m_model_path,
                                                              cache_location)) {
                    LOG_INFO_TAG("cache pre-compilation skipped for device: %s, cache blob already exists",
                                 device.device_name.c_str());
                } else {
                    auto compiled = model ? core->compile_model(model, device.device_name, config)
                                          : core->compile_model(m_model_path, device.device_name, config);
                    LOG_INFO_TAG("cache pre-compilation finished for device: %s, compile time: %lf ms",
                                 device.device_name.c_str(),
                                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                                     .count());
                }
            }
        } catch (const std::exception& e) {
            LOG_WARNING_TAG("cache pre-compilation failed for device: %s, %s", device.device_name.c_str(), e.what());
        } catch (...) {
            LOG_WARNING_TAG("cache pre-compilation failed for device: %s, unknown exception",
                            device.device_name.c_str());
        }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_compiling = false;
        }
        m_cv.notify_all();
    }
}

CacheWarmupManager::~CacheWarmupManager() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
    }
    m_cv.notify_all();
    if (m_worker.joinable()) {
        m_worker.join();
    }
}

void CacheWarmupManager::enqueue(std::shared_ptr<CacheWarmup> warmup) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_worker.joinable()) {
            m_worker = std::thread([this] {
                worker_loop();
            });
        }
        m_queue.push_back(std::move(warmup));
    }
    m_cv.notify_one();
}

void CacheWarmupManager::worker_loop() {
    while (true) {
        std::shared_ptr<CacheWarmup> warmup;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this] {
                return m_stop || !m_queue.empty();
            });
            if (m_stop) {
                return;
            }
            warmup = std::move(m_queue.front());
            m_queue.pop_front();
        }
        warmup->run();
    }
}

std::filesystem::path CacheWarmupManager::get_cache_location(const ov::ICore& core,
                                                             const ov::AnyMap& config,
                                                             const std::string& device_name) {
    if (const auto it = config.find(ov::cache_dir.name()); it != config.end()) {
        return ov::util::make_path(it->second.as<std::string>());
    }
    if (const auto it = config.find(ov::cache_path.name()); it != config.end()) {
        return it->second.as<std::filesystem::path>();
    }

    const auto query = [&core](const std::string& scope) -> std::optional<std::filesystem::path> {
        try {
            if (const auto cache_dir = core.get_property(scope, ov::cache_dir); !cache_dir.empty()) {
                return ov::util::make_path(cache_dir);
            }
            return std::filesystem::path(core.get_property(scope, ov::cache_path));
        } catch (const std::exception&) {
            return std::nullopt;
        }
    };
    if (!device_name.empty()) {
        if (auto location = query(device_name)) {
            return *location;
        }
    }
    return query("").value_or(std::filesystem::path{});
}

bool CacheWarmupManager::is_blob_cached(const ov::ICore& core,
                                        const std::string& device_name,
                                        const ov::AnyMap& config,
                                        const std::shared_ptr<const ov::Model>& model,
                                        const std::filesystem::path& model_path,
                                        const std::filesystem::path& cache_location) {
    // A single-file cache (*.bin) cannot be probed per blob, so treat it as a miss.
    if (cache_location.empty() || cache_location.extension() == ".bin" || (!model && model_path.empty())) {
        return false;
    }
    const auto base_name = ov::DeviceIDParser(device_name).get_device_name();
    const auto compile_config =
        core.create_compile_config(base_name, core.get_supported_property(base_name, config));
    const auto blob_id = model ? ov::ModelCache::compute_hash(model, compile_config)
                               : ov::ModelCache::compute_hash(model_path, compile_config);
    return ov::util::file_exists(cache_location / (blob_id + ".blob"));
}

}  // namespace auto_plugin
}  // namespace ov
