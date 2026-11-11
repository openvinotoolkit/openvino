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

// Cache warm-up state of one compiled model, used when ov::intel_auto::compile_for_all is enabled. While AUTO serves
// requests on the actual device, the other candidate devices are compiled in the background so that their cache blobs
// exist the next time the model is loaded (e.g. when a different device is picked later). Needs a cache dir/path.
//
// Flow:
//  1. start() is called once the actual device is known. It filters out the actual device, then hands this object
//     to CacheWarmupManager (once per object).
//  2. The manager's single worker thread calls run(), which handles the devices one by one:
//     - caching disabled for the device (no cache dir/path) -> skip;
//     - otherwise call core->compile_model(): Core imports a valid cached blob or compiles and writes a new one;
//       the resulting compiled model is dropped immediately. The host load is reduced: GPU gets low host priority,
//       and the compile threads are limited to hardware threads - 2 (at least 1) via compilation_num_threads, or
//       inference_num_threads for CPU. A config change that would alter the cache key is not applied.
//     A failure on one device is logged and does not affect the other devices or the user's compiled model.
//  3. stop() (owning compiled model destroyed) skips the not yet started devices and waits for the compilation in
//     flight, so the core and the model are not used after the owner is gone.
//
// State transitions (guarded by m_mutex):
//
//   Created --start(): nothing to warm up--> Done
//      |
//      +--start(): devices queued--> Idle <--------------------+
//                                      |                        | compile finished
//                                      +--run(): device--> Compiling
//                                      |                        |
//                                      | all devices done       | stop()
//                                      v                        v
//                                    Done                   Stopping --compile finished--> Stopped
//
//   stop() in Created, Idle or Done goes straight to Stopped; start() is accepted only in Created.
class CacheWarmup : public std::enable_shared_from_this<CacheWarmup> {
public:
    CacheWarmup(CacheWarmupManager& manager,
                std::weak_ptr<ov::ICore> core,
                std::shared_ptr<const ov::Model> model,
                std::filesystem::path model_path,
                std::string log_tag);

    // Queues the warm-up of the candidates other than the actual device; only the first call counts.
    void start(const std::string& actual_device, const std::vector<DeviceInformation>& candidates);
    // Skips the remaining devices and waits for the in-flight compilation; idempotent.
    void stop();
    // Compiles the queued devices one by one; called by the CacheWarmupManager worker thread only.
    void run();

private:
    void request_stop_unsafe();
    const std::string& get_log_tag() const noexcept;

    CacheWarmupManager& m_manager;
    // The owning compiled model keeps the core alive until stop() returns.
    std::weak_ptr<ov::ICore> m_core;
    std::shared_ptr<const ov::Model> m_model;
    std::filesystem::path m_model_path;
    std::string m_log_tag;
    std::vector<DeviceInformation> m_devices;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    enum class State {
        Created,    // start() has not been called
        Idle,       // queued, or between two device compilations
        Compiling,  // a device compilation is in flight
        Stopping,   // stop() arrived during a compilation; run() completes the stop
        Stopped,    // stop() completed; nothing will be compiled anymore
        Done        // all devices processed, or there was nothing to warm up
    };
    State m_state = State::Created;
};

// Plugin-wide queue; a single worker thread serializes the warm-up compilations of all compiled models.
class CacheWarmupManager {
public:
    CacheWarmupManager() = default;
    CacheWarmupManager(const CacheWarmupManager&) = delete;
    CacheWarmupManager& operator=(const CacheWarmupManager&) = delete;
    CacheWarmupManager(CacheWarmupManager&&) = delete;
    CacheWarmupManager& operator=(CacheWarmupManager&&) = delete;
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
