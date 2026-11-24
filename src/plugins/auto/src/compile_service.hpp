// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "cache_manager.hpp"
#include "common.hpp"

namespace ov {
namespace auto_plugin {

// What the compilations of one device cost, used to decide whether its model may be released.
struct CompileStats {
    double last_compile_ms = 0.0;
    // empty if the device does not report ov::loaded_from_cache
    std::optional<bool> last_loaded_from_cache;
    size_t compile_count = 0;
    // the config of the last compilation, the input of the lazy blob ID
    ov::AnyMap last_config;
    // the cache blob ID of the last compilation, set by CompileService::blob_id()
    std::optional<std::string> last_blob_id;
};

/**
 * @brief The single entry point of a schedule to compile the model on a device.
 *
 * On-demand compilations run on the caller's thread. With ov::intel_auto::compile_for_all, the other candidates are
 * compiled in a background lane (CachePrecompiler) only to leave their cache blobs. An on-demand compilation of a
 * device first waits for an in-flight background compilation of it and keeps it from being compiled in the background
 * afterwards, so a device is never compiled twice at the same time and the on-demand compilation loads the blob.
 */
class CompileService {
public:
    CompileService(std::shared_ptr<ov::ICore> core, std::filesystem::path model_path, std::string log_tag);
    CompileService(const CompileService&) = delete;
    CompileService& operator=(const CompileService&) = delete;
    ~CompileService();

    // throws what Core throws; the model is ignored when the schedule was created from a model path
    SoCompiledModel compile(const std::shared_ptr<const ov::Model>& model,
                            const DeviceName& device,
                            const ov::AnyMap& config);
    std::optional<CompileStats> stats(const DeviceName& device) const;
    // The cache blob ID of the last compilation of the device, computed on first use because hashing the model is
    // costly; empty if the device does not reload its model from a cache. The model is ignored for a model path.
    std::optional<std::string> blob_id(const DeviceName& device, const std::shared_ptr<const ov::Model>& model);

    // Whether the cache of the device holds the blob of its last compilation: the export may have failed and the device
    // reports nothing about it, so the blob itself is looked up. A hit is remembered, a blob never disappears.
    bool has_cache_blob(const DeviceName& device, const std::shared_ptr<const ov::Model>& model);

    // sets up the background lane; must be called before the startup compilation releases the model
    void enable_precompile(CacheManager& manager, std::shared_ptr<const ov::Model> model);
    bool has_precompile() const;
    void start_precompile(const DeviceName& actual_device,
                          const std::vector<DeviceInformation>& candidates,
                          CachePrecompiler::DeviceLoadedCheck is_device_loaded);
    // skips the queued devices and waits for the in-flight background compilation; idempotent
    void stop_precompile();

private:
    const std::string& get_log_tag() const noexcept;

    std::shared_ptr<ov::ICore> m_core;
    std::filesystem::path m_model_path;
    std::string m_log_tag;
    // set before any compilation starts, never reset afterwards
    std::shared_ptr<CachePrecompiler> m_precompiler;
    mutable std::mutex m_stats_mutex;
    std::map<DeviceName, CompileStats> m_stats;
};

}  // namespace auto_plugin
}  // namespace ov
