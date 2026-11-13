// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cache_manager.hpp"

#include <algorithm>
#include <chrono>
#include <map>
#include <optional>

#include "openvino/runtime/compilation_context.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/util/container_util.hpp"
#include "openvino/util/file_util.hpp"

namespace ov {
namespace auto_plugin {
namespace {
// Core keeps all blobs of a *.bin cache path in one archive. Its writes are only locked per blob ID, so a background
// pre-compilation must not write into it concurrently with other compilations.
bool is_single_file_cache(const std::filesystem::path& cache_location) {
    std::error_code ec;
    const auto status = std::filesystem::status(cache_location, ec);
    if (!ec && std::filesystem::exists(status)) {
        return !std::filesystem::is_directory(status);
    }
    // Core creates the cache directory on the first write and picks the storage by extension until then.
    return cache_location.extension() == ".bin";
}

// Core moves the ID of e.g. "GPU.1" into ov::device::id, so the cache key must be built for that device instance.
ov::AnyMap create_compile_config(const ov::ICore& core, const std::string& device_name, const ov::AnyMap& config) {
    const ov::DeviceIDParser parser(device_name);
    auto device_config = config;
    if (!parser.get_device_id().empty()) {
        device_config.emplace(ov::device::id.name(), parser.get_device_id());
    }
    return core.create_compile_config(parser.get_device_name(), device_config);
}

// Keys are compared as strings, like the cache hash does: ov::Any::operator== throws for values such as ov::device::UUID.
std::map<std::string, std::string> get_cache_key(const ov::ICore& core,
                                                 const std::string& device_name,
                                                 const ov::AnyMap& config) {
    std::map<std::string, std::string> key;
    for (const auto& [name, value] : create_compile_config(core, device_name, config)) {
        key.emplace(name, value.as<std::string>());
    }
    return key;
}

// Lowers the host-side cost of the background compilation without changing the cache key.
ov::AnyMap make_precompile_config(const ov::ICore& core, const DeviceInformation& device) {
    std::vector<ov::PropertyName> supported;
    try {
        supported = core.get_property(device.device_name, ov::supported_properties);
    } catch (const ov::Exception&) {
        return device.config;
    }
    auto config = device.config;
    if (ov::util::contains(supported, ov::intel_gpu::hint::host_task_priority.name())) {
        config.emplace(ov::intel_gpu::hint::host_task_priority(ov::hint::Priority::LOW));
    }
    if (ov::util::contains(supported, ov::compilation_num_threads.name())) {
        config.emplace(ov::compilation_num_threads(std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 4)));
    }
    if (config.size() == device.config.size()) {
        return config;
    }
    if (get_cache_key(core, device.device_name, config) != get_cache_key(core, device.device_name, device.config)) {
        return device.config;
    }
    return config;
}
}  // namespace

// The caller may modify its model once compile_model() returns, so pre-compilation works on a private clone; a
// clone failure propagates to the caller.
CachePrecompiler::CachePrecompiler(CacheManager& manager,
                                   std::weak_ptr<ov::ICore> core,
                                   std::shared_ptr<const ov::Model> model,
                                   std::filesystem::path model_path,
                                   std::string log_tag)
    : m_manager(manager),
      m_core(std::move(core)),
      m_model(model ? model->clone() : nullptr),
      m_model_path(std::move(model_path)),
      m_log_tag(std::move(log_tag)) {}

const std::string& CachePrecompiler::get_log_tag() const noexcept {
    return m_log_tag;
}

void CachePrecompiler::start(const std::string& actual_device, const std::vector<DeviceInformation>& candidates) {
    std::vector<DeviceInformation> devices;
    for (const auto& device : candidates) {
        if (device.device_name != actual_device && ov::DeviceIDParser(device.device_name).get_device_name() != "CPU") {
            devices.push_back(device);
        }
    }
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_state != State::Created) {
            return;
        }
        if (devices.empty() || (!m_model && m_model_path.empty())) {
            m_model.reset();
            m_state = State::Done;
            return;
        }
        m_state = State::Idle;
        m_devices = std::move(devices);
        LOG_INFO_TAG("start cache pre-compilation for %lu device(s)", static_cast<unsigned long>(m_devices.size()));
    }
    m_manager.enqueue(shared_from_this());
}

void CachePrecompiler::request_stop_unsafe() {
    m_model.reset();
    if (m_state == State::Compiling) {
        m_state = State::Stopping;
    } else if (m_state != State::Stopping) {
        m_state = State::Stopped;
    }
}

void CachePrecompiler::stop() {
    std::unique_lock<std::mutex> lock(m_mutex);
    request_stop_unsafe();
    m_cv.wait(lock, [this] {
        return m_state != State::Stopping;
    });
}

void CachePrecompiler::run() {
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
            if (m_state != State::Idle) {
                return;
            }
            m_state = State::Compiling;
        }
        const auto begin = std::chrono::steady_clock::now();
        try {
            if (auto core = m_core.lock()) {
                const auto config = make_precompile_config(*core, device);
                const auto cache_location =
                    CacheManager::get_cache_location(*core, config, device.device_name);
                if (cache_location.empty()) {
                    LOG_INFO_TAG("cache pre-compilation skipped for device: %s, caching is disabled for it",
                                 device.device_name.c_str());
                } else if (!CacheManager::device_supports_caching(*core, device.device_name)) {
                    LOG_INFO_TAG("cache pre-compilation skipped for device: %s, it does not support model caching",
                                 device.device_name.c_str());
                } else if (is_single_file_cache(cache_location)) {
                    LOG_WARNING_TAG("cache pre-compilation skipped for device: %s, single-file cache %s is not "
                                    "supported",
                                    device.device_name.c_str(),
                                    cache_location.string().c_str());
                } else {
                    // The compiled model is dropped on purpose: Core writes the cache blob as a side effect.
                    const auto compiled_model = model ? core->compile_model(model, device.device_name, config)
                                                      : core->compile_model(m_model_path, device.device_name, config);
                    const char* loaded_from_cache = "unknown";
                    try {
                        loaded_from_cache =
                            compiled_model->get_property(ov::loaded_from_cache.name()).as<bool>() ? "yes" : "no";
                    } catch (const ov::Exception&) {
                    }
                    LOG_INFO_TAG("cache pre-compilation finished for device: %s, compile time: %lf ms, loaded from "
                                 "cache: %s",
                                 device.device_name.c_str(),
                                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                                     .count(),
                                 loaded_from_cache);
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
            m_state = m_state == State::Stopping ? State::Stopped : State::Idle;
        }
        m_cv.notify_all();
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_state == State::Idle) {
        m_state = State::Done;
    }
}

CacheManager::~CacheManager() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
    }
    m_cv.notify_all();
    if (m_worker.joinable()) {
        m_worker.join();
    }
}

void CacheManager::enqueue(std::shared_ptr<CachePrecompiler> precompile) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_worker.joinable()) {
            m_worker = std::thread([this] {
                worker_loop();
            });
        }
        m_queue.push_back(std::move(precompile));
    }
    m_cv.notify_one();
}

void CacheManager::worker_loop() {
    while (true) {
        std::shared_ptr<CachePrecompiler> precompile;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this] {
                return m_stop || !m_queue.empty();
            });
            if (m_stop) {
                return;
            }
            precompile = std::move(m_queue.front());
            m_queue.pop_front();
        }
        precompile->run();
    }
}

std::filesystem::path CacheManager::get_cache_location(const ov::ICore& core,
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

std::string CacheManager::get_blob_id(const ov::ICore& core,
                                      const std::string& device_name,
                                      const ov::AnyMap& config,
                                      const std::shared_ptr<const ov::Model>& model,
                                      const std::filesystem::path& model_path) {
    // Same precedence as Core: a user blob ID wins, then the hash (which covers ov::cache_model_path for a model).
    if (const auto it = config.find(ov::cache_blob_id.name()); it != config.end()) {
        return it->second.as<std::string>();
    }
    OPENVINO_ASSERT(model || !model_path.empty(), "A model or model path is required to compute a cache blob ID");
    const auto compile_config =
        create_compile_config(core, device_name, core.get_supported_property(device_name, config));
    if (model) {
        const auto it = config.find(ov::cache_model_path.name());
        const auto cache_model_path =
            it == config.end() ? std::filesystem::path{} : it->second.as<std::filesystem::path>();
        return ov::ModelCache::compute_hash(model, cache_model_path, compile_config);
    }
    return ov::ModelCache::compute_hash(model_path, compile_config);
}

bool CacheManager::is_blob_cached(const std::filesystem::path& cache_location, const std::string& blob_id) {
    // The cache location is a directory with one <blob_id>.blob per model; a single-file cache is never a hit here.
    return !cache_location.empty() && ov::util::file_exists(cache_location / (blob_id + ".blob"));
}

bool CacheManager::is_blob_cached(const ov::ICore& core,
                                  const std::string& device_name,
                                  const ov::AnyMap& config,
                                  const std::shared_ptr<const ov::Model>& model,
                                  const std::filesystem::path& model_path) {
    const auto cache_location = get_cache_location(core, config, device_name);
    if (cache_location.empty() || (!model && model_path.empty()) || !device_supports_caching(core, device_name)) {
        return false;
    }
    return is_blob_cached(cache_location, get_blob_id(core, device_name, config, model, model_path));
}

bool CacheManager::is_reloadable(const ov::ICore& core,
                                 const std::string& device_name,
                                 DeviceCacheInfo& info,
                                 const std::shared_ptr<const ov::Model>& model,
                                 const std::filesystem::path& model_path) {
    const ov::AnyMap no_config;
    const auto& config = info.compile_config ? *info.compile_config : no_config;
    if (!info.supports_caching) {
        info.supports_caching = device_supports_caching(core, device_name);
    }
    const auto cache_location = get_cache_location(core, config, device_name);
    if (!*info.supports_caching || cache_location.empty()) {
        return false;
    }
    if (!info.blob_id) {
        info.blob_id = get_blob_id(core, device_name, config, model, model_path);
    }
    // The cache entry may have been removed after the blob ID was computed, so the file is checked every time.
    return is_blob_cached(cache_location, *info.blob_id);
}

bool CacheManager::disable_caching(const ov::ICore& core, const std::string& device_name, ov::AnyMap& config) {
    if (get_cache_location(core, config, device_name).empty()) {
        return false;
    }
    config.erase(ov::cache_path.name());
    config[ov::cache_dir.name()] = std::string{};
    return true;
}

bool CacheManager::device_supports_caching(const ov::ICore& core, const std::string& device_name) {
    try {
        return core.device_supports_model_caching(device_name);
    } catch (const ov::Exception&) {
        return false;
    }
}

bool CacheManager::has_user_blob_id(const ov::AnyMap& config) {
    return config.count(ov::cache_blob_id.name()) != 0;
}

}  // namespace auto_plugin
}  // namespace ov
