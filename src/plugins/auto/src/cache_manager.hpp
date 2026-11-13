// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "common.hpp"

namespace ov {
namespace auto_plugin {

class CacheManager;

// What AUTO has learned about the model cache of a device, each value is queried at most once.
struct DeviceCacheInfo {
    std::optional<bool> supports_caching;
    std::optional<std::string> blob_id;
    // the config the model was compiled with, it decides the cache key
    std::optional<ov::AnyMap> compile_config;
};

// Pre-compilation state of one compiled model, used when ov::intel_auto::compile_for_all is enabled. While AUTO serves
// requests on the actual device, the other candidate devices are compiled in the background so that their cache blobs
// exist the next time the model is loaded (e.g. when a different device is picked later). Needs a cache dir/path.
//
// Flow:
//  1. start() is called once the actual device is known. It filters out the actual device and CPU, then hands this
//     object to CacheManager (once per object).
//  2. The manager's single worker thread calls run(), which handles the devices one by one:
//     - caching disabled for the device (no cache dir/path) -> skip;
//     - the device does not support model caching -> skip, compiling it would write no blob;
//     - a single-file cache (*.bin) -> skip with a warning, its writes are not safe to run in the background;
//     - an existing blob is not probed for: Core decides whether it is still usable, so a stale one is rewritten;
//     - otherwise call core->compile_model() with a low-priority, reduced-thread config that keeps the cache key
//       unchanged, so Core itself writes the blob; the resulting compiled model is dropped immediately.
//     A failure on one device is logged and does not affect the other devices or the user's compiled model.
//  3. stop() (owning compiled model destroyed) skips the not yet started devices and waits for the compilation in
//     flight, so the core and the model are not used after the owner is gone.
//
// State transitions (guarded by m_mutex):
//
//   Created --start(): nothing to pre-compile--> Done
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
class CachePrecompiler : public std::enable_shared_from_this<CachePrecompiler> {
public:
    CachePrecompiler(CacheManager& manager,
                     std::weak_ptr<ov::ICore> core,
                     std::shared_ptr<const ov::Model> model,
                     std::filesystem::path model_path,
                     std::string log_tag);

    // Queues the pre-compilation of the candidates other than the actual device; only the first call counts.
    void start(const std::string& actual_device, const std::vector<DeviceInformation>& candidates);
    // Skips the remaining devices and waits for the in-flight compilation; idempotent.
    void stop();
    // Compiles the queued devices one by one; called by the CacheManager worker thread only.
    void run();

private:
    void request_stop_unsafe();
    const std::string& get_log_tag() const noexcept;

    CacheManager& m_manager;
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
        Done        // all devices processed, or there was nothing to pre-compile
    };
    State m_state = State::Created;
};

// Device cache management of the AUTO plugin: static helpers that locate and probe the cache, and a plugin-wide queue
// whose single worker thread serializes the pre-compilations of all compiled models.
class CacheManager {
public:
    CacheManager() = default;
    CacheManager(const CacheManager&) = delete;
    CacheManager& operator=(const CacheManager&) = delete;
    CacheManager(CacheManager&&) = delete;
    CacheManager& operator=(CacheManager&&) = delete;
    ~CacheManager();

    void enqueue(std::shared_ptr<CachePrecompiler> precompile);

    // Cache location resolved like the Core does: ov::cache_dir / ov::cache_path in config, then the setting of the
    // device (when given), then the global one. An empty result means caching is disabled.
    static std::filesystem::path get_cache_location(const ov::ICore& core,
                                                    const ov::AnyMap& config,
                                                    const std::string& device_name = {});
    static std::string get_blob_id(const ov::ICore& core,
                                   const std::string& device_name,
                                   const ov::AnyMap& config,
                                   const std::shared_ptr<const ov::Model>& model,
                                   const std::filesystem::path& model_path);
    static bool is_blob_cached(const std::filesystem::path& cache_location, const std::string& blob_id);
    static bool is_blob_cached(const ov::ICore& core,
                               const std::string& device_name,
                               const ov::AnyMap& config,
                               const std::shared_ptr<const ov::Model>& model,
                               const std::filesystem::path& model_path);
    // Whether a released compiled model can be imported from the cache again. The first call queries and memoizes
    // supports_caching and blob_id in info, the blob file itself is checked on every call.
    static bool is_reloadable(const ov::ICore& core,
                              const std::string& device_name,
                              DeviceCacheInfo& info,
                              const std::shared_ptr<const ov::Model>& model,
                              const std::filesystem::path& model_path);
    // Makes the device skip the cache: Core rejects a config that holds both ov::cache_dir and ov::cache_path.
    // Returns false if the device would not use a cache anyway.
    static bool disable_caching(const ov::ICore& core, const std::string& device_name, ov::AnyMap& config);
    // Whether Core writes a cache blob for the device; false if the query fails.
    static bool device_supports_caching(const ov::ICore& core, const std::string& device_name);
    // A user blob ID does not identify a device or a model, so it can not be probed or pre-compiled per device.
    static bool has_user_blob_id(const ov::AnyMap& config);

private:
    void worker_loop();

    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::deque<std::shared_ptr<CachePrecompiler>> m_queue;
    bool m_stop = false;
    std::thread m_worker;
    std::shared_ptr<Log> m_logger = Log::instance();
};

}  // namespace auto_plugin
}  // namespace ov
