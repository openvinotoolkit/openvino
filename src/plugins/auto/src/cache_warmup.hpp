// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common.hpp"

namespace ov {
namespace auto_plugin {

class CacheWarmupManager;

// Cache warm-up state of one compiled model: compiles the model to the given devices in the background
// to populate cache blobs, releasing each compiled model right away.
class CacheWarmup : public std::enable_shared_from_this<CacheWarmup> {
public:
    CacheWarmup(CacheWarmupManager& manager,
                std::weak_ptr<ov::ICore> core,
                std::shared_ptr<const ov::Model> model,
                std::filesystem::path model_path,
                std::string log_tag);

    // Queues the warm-up of the candidates other than the actual device and CPU; only the first call counts.
    void start(const std::string& actual_device, const std::vector<DeviceInformation>& candidates);
    // Skips the remaining devices and waits for the in-flight compilation; idempotent.
    void stop();

private:
    friend class CacheWarmupManager;
    void run();
    std::string get_log_tag() const noexcept;

    CacheWarmupManager& m_manager;
    // The owning compiled model keeps the core alive until stop() returns.
    std::weak_ptr<ov::ICore> m_core;
    std::shared_ptr<const ov::Model> m_model;
    std::filesystem::path m_model_path;
    std::string m_log_tag;
    std::vector<DeviceInformation> m_devices;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_started = false;
    bool m_stopped = false;
    bool m_compiling = false;
};

// Plugin-wide queue; a single worker thread serializes the warm-up compilations of all compiled models.
class CacheWarmupManager {
public:
    CacheWarmupManager() = default;
    CacheWarmupManager(const CacheWarmupManager&) = delete;
    CacheWarmupManager& operator=(const CacheWarmupManager&) = delete;
    ~CacheWarmupManager();

    void enqueue(std::shared_ptr<CacheWarmup> warmup);

    // Cache location resolved like the Core does: ov::cache_dir / ov::cache_path in config, then the setting of the
    // device (when given), then the global one. An empty result means caching is disabled.
    static std::filesystem::path get_cache_location(const ov::ICore& core,
                                                    const ov::AnyMap& config,
                                                    const std::string& device_name = {});
    static bool is_blob_cached(const ov::ICore& core,
                               const std::string& device_name,
                               const ov::AnyMap& config,
                               const std::shared_ptr<const ov::Model>& model,
                               const std::filesystem::path& model_path,
                               const std::filesystem::path& cache_location);

private:
    void worker_loop();

    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::deque<std::shared_ptr<CacheWarmup>> m_queue;
    bool m_stop = false;
    std::thread m_worker;
    std::shared_ptr<Log> m_logger = Log::instance();
};

}  // namespace auto_plugin
}  // namespace ov
