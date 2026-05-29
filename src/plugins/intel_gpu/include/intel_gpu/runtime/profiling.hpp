// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <atomic>
#include <mutex>
#include "intel_gpu/runtime/execution_config.hpp"
#include "openvino/runtime/properties.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef NOGDI
#define NOGDI
#endif

#include <windows.h>
#include "psapi.h"
#include <dxgi1_4.h>
#include <wrl/client.h>
#pragma comment(lib, "dxgi.lib")
#endif

#include "layout.hpp"
#include "utils.hpp"
#include "debug_configuration.hpp"

namespace cldnn {
namespace instrumentation {
/// @addtogroup cpp_api C++ API
/// @{

/// @addtogroup cpp_event Events Support
/// @{

/// @brief Represents profiling intervals stages.
enum class profiling_stage {
    submission,  // Time spent on submitting command by the host to the device associated with the commandqueue.
    starting,    // Time spent on waiting in the commandqueue before execution.
    executing,   // Time spent on command execution.
    duration     // Time spent on command execution for CPU layers.
};

/// @brief Helper class to calculate time periods.
template <class ClockTy = std::chrono::steady_clock>
class timer {
    typename ClockTy::time_point start_point;

public:
    /// @brief Timer value type.
    typedef typename ClockTy::duration val_type;

    /// @brief Starts timer.
    timer() : start_point(ClockTy::now()) {}

    /// @brief Returns time eapsed since construction.
    val_type uptime() const { return ClockTy::now() - start_point; }
};

/// @brief Abstract class to represent profiling period.
struct profiling_period {
    /// @brief Returns profiling period value.
    virtual std::chrono::nanoseconds value() const = 0;
    /// @brief Destructor.
    virtual ~profiling_period() = default;
};

/// @brief Basic @ref profiling_period implementation which stores data as an simple period value.
struct profiling_period_basic : profiling_period {
    /// @brief Constructs from @p std::chrono::duration.
    template <class _Rep, class _Period>
    explicit profiling_period_basic(const std::chrono::duration<_Rep, _Period>& val)
        : _value(std::chrono::duration_cast<std::chrono::nanoseconds>(val)) {}

    /// @brief Returns profiling period value passed in constructor.
    std::chrono::nanoseconds value() const override { return _value; }

private:
    std::chrono::nanoseconds _value;
};

/// @brief Represents profiling interval as its type and value.
struct profiling_interval {
    profiling_stage stage;                    ///< @brief Display name.
    std::shared_ptr<profiling_period> value;  ///< @brief Interval value.
};

/// @brief Represents list of @ref profiling_interval
struct profiling_info {
    std::string name;                           ///< @brief Display name.
    std::vector<profiling_interval> intervals;  ///< @brief List of intervals.
};

enum class pipeline_stage : uint8_t {
    shape_inference = 0,
    update_implementation = 1,
    update_weights = 2,
    memory_allocation = 3,
    set_arguments = 4,
    inference = 5
};

inline std::ostream& operator<<(std::ostream& os, const pipeline_stage& stage) {
    switch (stage) {
        case pipeline_stage::shape_inference:       return os << "shape_inference";
        case pipeline_stage::update_implementation: return os << "update_implementation";
        case pipeline_stage::set_arguments:         return os << "set_arguments";
        case pipeline_stage::update_weights:        return os << "update_weights";
        case pipeline_stage::memory_allocation:     return os << "memory_allocation";
        case pipeline_stage::inference:             return os << "inference";
        default: OPENVINO_ASSERT(false, "[GPU] Unexpected pipeline stage");
    }
}

struct perf_counter_key {
    std::vector<layout> network_input_layouts;
    std::vector<layout> input_layouts;
    std::vector<layout> output_layouts;
    std::string impl_name;
    pipeline_stage stage;
    int64_t iteration_num;
    bool cache_hit;
    std::string memalloc_info;
};

struct perf_counter_hash {
    std::size_t operator()(const perf_counter_key& k) const {
        size_t seed = 0;
        seed = hash_combine(seed, static_cast<std::underlying_type<instrumentation::pipeline_stage>::type>(k.stage));
        seed = hash_combine(seed, static_cast<int>(k.cache_hit));
        seed = hash_combine(seed, k.iteration_num);
        for (auto& layout : k.network_input_layouts) {
            for (auto& d : layout.get_shape()) {
                seed = hash_combine(seed, d);
            }
        }
        for (auto& layout : k.input_layouts) {
            for (auto& d : layout.get_shape()) {
                seed = hash_combine(seed, d);
            }
        }
        for (auto& layout : k.output_layouts) {
            for (auto& d : layout.get_shape()) {
                seed = hash_combine(seed, d);
            }
        }
        return seed;
    }
};

template<typename ProfiledObjectType>
class profiled_stage {
public:
    profiled_stage(bool profiling_enabled, ProfiledObjectType& obj, instrumentation::pipeline_stage stage)
        : profiling_enabled(profiling_enabled)
        , _obj(obj)
        , _stage(stage) {
        GPU_DEBUG_IF(profiling_enabled) {
            _per_iter_mode = GPU_DEBUG_VALUE_OR(ov::intel_gpu::ExecutionConfig::get_dump_profiling_data_per_iter(), false);
            _start = std::chrono::high_resolution_clock::now();
        }
    }

    ~profiled_stage() {
        GPU_DEBUG_IF(profiling_enabled) {
            using us = std::chrono::microseconds;

            _finish = std::chrono::high_resolution_clock::now();
            auto stage_duration = std::chrono::duration_cast<us>(_finish - _start).count();
            auto custom_stage_duration = std::chrono::duration_cast<us>(custom_duration).count();
            auto total_duration = custom_stage_duration == 0 ? stage_duration
                                                             : custom_stage_duration;
            _obj.add_profiling_data(_stage, cache_hit, memalloc_info, total_duration, _per_iter_mode);
        }
    }
    void set_cache_hit(bool val = true) { cache_hit = val; }
    void add_memalloc_info(std::string info = "") { memalloc_info += info; }
    void set_custom_stage_duration(std::chrono::nanoseconds duration) { custom_duration = duration; }

private:
    bool profiling_enabled = false;
    std::chrono::high_resolution_clock::time_point _start = {};
    std::chrono::high_resolution_clock::time_point _finish = {};
    std::chrono::nanoseconds custom_duration = {};
    ProfiledObjectType& _obj;
    instrumentation::pipeline_stage _stage;
    bool _per_iter_mode = false;
    bool cache_hit = false;
    std::string memalloc_info = "";
};

class mem_usage_logger {
public:
    struct memory_footprint {
        memory_footprint() : rss(0), peak_rss(0) {}
        memory_footprint(int64_t rss, int64_t peak_rss) : rss(rss), peak_rss(peak_rss) {}
        int64_t rss;
        int64_t peak_rss;
    };

    mem_usage_logger(const std::string& stage_name, bool lifetime_logging_mode = true, bool print_mem_usage = true)
        : _stage_name(stage_name)
        , _lifetime_logging_mode(lifetime_logging_mode)
        , _print_mem_usage(print_mem_usage) {
        if (_lifetime_logging_mode)
            start_logging();
    }

    ~mem_usage_logger() {
        if (_lifetime_logging_mode)
            stop_logging();
        if (_print_mem_usage && _is_active)
            print_mem_usage_info();
    }

    void start_logging() {
        _is_active = true;
        _before = get_memory_footprint();
    }

    void stop_logging() {
        _after = get_memory_footprint();
    }

    memory_footprint get_elapsed_mem_usage() {
        return memory_footprint{ _after.rss - _before.rss, _after.peak_rss - _before.peak_rss };
    }

#if defined(_WIN32)
    static void set_active_luid(const ov::device::LUID& luid) {
        std::lock_guard<std::mutex> lock(get_luid_mutex());
        get_active_luid_storage() = luid;
    }
#endif

    void print_mem_usage_info() {
        auto mem_usage = get_elapsed_mem_usage();
        std::string log_msg = "Memory usage for " + _stage_name + ": " + std::to_string(mem_usage.rss) +
                              " KB (current RSS: " + std::to_string(_after.rss) +
                              " KB; peak RSS: " + std::to_string(_after.peak_rss) + " KB)";

        int64_t gpu_vram_used = get_gpu_dedicated_vram_used_kb();
        if (gpu_vram_used >= 0) {
            log_msg += ", (gpu_vram_used : " + std::to_string(gpu_vram_used) + " KB)";
        }

        GPU_DEBUG_LOG << log_msg << std::endl;
    }

    int64_t get_gpu_dedicated_vram_used_kb() {
#if defined(_WIN32)
        using Microsoft::WRL::ComPtr;
        static ComPtr<IDXGIFactory4> dxgi_factory;
        static ComPtr<IDXGIAdapter3> selected_adapter;
        static bool initialized = false;
        static bool init_failed = false;
        static ov::device::LUID selected_luid{};

        ov::device::LUID active_luid;
        {
            std::lock_guard<std::mutex> lock(get_luid_mutex());
            active_luid = get_active_luid_storage();
        }

        if (is_luid_empty(active_luid)) {
            return -1;
        }

        if (initialized && selected_luid.luid != active_luid.luid) {
            selected_adapter.Reset();
            initialized = false;
            init_failed = false;
        }

        if (!initialized && !init_failed) {
            // One-time initialization of DXGI
            if (S_OK != CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgi_factory))) {
                init_failed = true;
                return -1;
            }

            IDXGIAdapter *adapter = nullptr;
            bool found_intel_adapter = false;

            for (UINT adapterIndex = 0; dxgi_factory->EnumAdapters(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND; adapterIndex++) {
                DXGI_ADAPTER_DESC desc;
                adapter->GetDesc(&desc);
                if (match_luid(desc.AdapterLuid, active_luid)) {
                    if (S_OK == adapter->QueryInterface(IID_PPV_ARGS(&selected_adapter))) {
                        found_intel_adapter = true;
                    }
                    adapter->Release();
                    if (found_intel_adapter)
                        break;
                }
                adapter->Release();
            }

            if (!found_intel_adapter) {
                init_failed = true;
                return -1;
            }
            initialized = true;
            selected_luid = active_luid;
        }

        if (init_failed || !selected_adapter) {
            return -1;
        }

        // Query VRAM usage across all nodes
        int64_t total_vram_used = 0;
        const int KiB = 1024;

        int nodeId = 0;
        DXGI_QUERY_VIDEO_MEMORY_INFO info;
        while (S_OK == selected_adapter->QueryVideoMemoryInfo(nodeId, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)) {
            total_vram_used += info.CurrentUsage / KiB;
            nodeId++;
        }

        return total_vram_used;
#else
        return -1;  // Not available on non-Windows platforms
#endif
    }

private:
#if defined(_WIN32)
    static ov::device::LUID& get_active_luid_storage() {
        static ov::device::LUID active_luid{};
        return active_luid;
    }

    static std::mutex& get_luid_mutex() {
        static std::mutex mtx;
        return mtx;
    }

    static bool is_luid_empty(const ov::device::LUID& luid) {
        for (auto b : luid.luid) {
            if (b != 0) return false;
        }
        return true;
    }

    static bool match_luid(const ::LUID& dxgi_luid, const ov::device::LUID& ov_luid) {
        // DXGI LUID: LowPart (4 bytes) + HighPart (4 bytes) = 8 bytes, same layout as ov::device::LUID
        return std::memcmp(&dxgi_luid, ov_luid.luid.data(), sizeof(dxgi_luid)) == 0;
    }
#endif

    memory_footprint get_memory_footprint() {
        memory_footprint footprint;
#if defined(_WIN32)
        PROCESS_MEMORY_COUNTERS pmc;
        GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
        footprint.rss = (int64_t)(pmc.WorkingSetSize/1024);
        footprint.peak_rss = (int64_t)(pmc.PeakWorkingSetSize/1024);
#elif !defined(__APPLE__)
        std::ifstream status("/proc/self/status");
        if (!status.is_open())
            return footprint;

        std::string line, title;
        while (std::getline(status, line)) {
            std::istringstream iss(line);
            iss >> title;
            if (title == "VmHWM:")
                iss >> footprint.peak_rss;
            else if (title == "VmRSS:")
                iss >> footprint.rss;
        }
#endif
        return footprint;
    }

    std::string _stage_name = {};
    bool _lifetime_logging_mode = false;
    bool _print_mem_usage = false;
    bool _is_active = false;
    memory_footprint _before;
    memory_footprint _after;
};

/// @}
/// @}
}  // namespace instrumentation
}  // namespace cldnn
