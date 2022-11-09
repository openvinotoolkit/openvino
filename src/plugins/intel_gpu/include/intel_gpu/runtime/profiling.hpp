// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <memory>
#include <vector>
#include <string>

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
    std::vector<layout> input_layouts;
    std::vector<layout> output_layouts;
    std::string impl_name;
    pipeline_stage stage;
    bool cache_hit;
};

struct perf_counter_hash {
    std::size_t operator()(const perf_counter_key& k) const {
        size_t seed = 0;
        seed = hash_combine(seed, static_cast<std::underlying_type<instrumentation::pipeline_stage>::type>(k.stage));
        seed = hash_combine(seed, static_cast<int>(k.cache_hit));
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
            _start = std::chrono::high_resolution_clock::now();
        }
    }

    ~profiled_stage() {
        GPU_DEBUG_IF(profiling_enabled) {
            _finish = std::chrono::high_resolution_clock::now();
            auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(_finish - _start).count();
            _obj.add_profiling_data(_stage, cache_hit, total_duration);
        }
    }
    void set_cache_hit(bool val = true) { cache_hit = val; }

private:
    bool profiling_enabled = false;
    std::chrono::high_resolution_clock::time_point _start = {};
    std::chrono::high_resolution_clock::time_point _finish = {};
    ProfiledObjectType& _obj;
    instrumentation::pipeline_stage _stage;
    bool cache_hit = false;
};

/// @}
/// @}
}  // namespace instrumentation
}  // namespace cldnn
