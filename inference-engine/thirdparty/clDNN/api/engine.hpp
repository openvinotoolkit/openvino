/*
// Copyright (c) 2016-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "cldnn.hpp"
#include "device.hpp"
#include <string>
#include <stdexcept>
#include <vector>
#include <map>

namespace cldnn {

/// @addtogroup cpp_api C++ API
/// @{

/// @defgroup cpp_engine Execution Engine
/// @{

/// @brief Defines available engine types
enum class engine_types : int32_t {
    ocl
};

/// @brief Defines available priority mode types
enum class priority_mode_types : int16_t {
    disabled,
    low,
    med,
    high
};

/// @brief Defines available priority mode types
enum class throttle_mode_types : int16_t {
    disabled,
    low,
    med,
    high
};

/// @brief Configuration parameters for created engine.
struct engine_configuration {
    const bool enable_profiling;              ///< Enable per-primitive profiling.
    const bool meaningful_kernels_names;      ///< Generate meaniful names fo OpenCL kernels.
    const bool dump_custom_program;           ///< Dump the user OpenCL programs to files
    const std::string compiler_options;       ///< OpenCL compiler options string.
    const std::string single_kernel_name;     ///< If provided, runs specific layer.
    const bool enable_parallelisation;        ///< Enables parallel execution of primitives which don't depend on each other. Disabled by default.
    const std::string engine_log;             ///< Specifies a file to which engine log should be dumped. Empty by default (means no logging).
    const std::string sources_dumps_dir;      ///< Specifies a directory where sources of cldnn::program objects should be dumped.
                                              ///< Empty by default (means no dumping).
    const priority_mode_types priority_mode;  ///< Priority mode (support of priority hints in command queue). If cl_khr_priority_hints extension
                                              ///< is not supported by current OpenCL implementation, the value must be set to cldnn_priority_disabled.

    const throttle_mode_types throttle_mode;  ///< Throttle mode (support of throttle hints in command queue). If cl_khr_throttle_hints extension
                                              ///< is not supported by current OpenCL implementation, the value must be set to cldnn_throttle_disabled.

    bool enable_memory_pool;              ///< Enables memory usage optimization. memory objects will be reused when possible
                                          ///< (switched off for older drivers then NEO).
    uint16_t n_streams;                   ///< Number of queues executed in parallel
    const std::string tuning_cache_path;  ///< Path to tuning kernel cache

    /// @brief Constructs engine configuration with specified options.
    /// @param profiling Enable per-primitive profiling.
    /// @param decorate_kernel_names Generate meaniful names fo OpenCL kernels.
    /// @param dump_custom_program Dump the custom OpenCL programs to files
    /// @param options OpenCL compiler options string.
    /// @param single_kernel If provided, runs specific layer.
    engine_configuration(
        bool profiling = false,
        bool decorate_kernel_names = false,
        bool dump_custom_program = false,
        const std::string& options = std::string(),
        const std::string& single_kernel = std::string(),
        bool primitives_parallelisation = true,
        const std::string& engine_log = std::string(),
        const std::string& sources_dumps_dir = std::string(),
        priority_mode_types priority_mode = priority_mode_types::disabled,
        throttle_mode_types throttle_mode = throttle_mode_types::disabled,
        bool memory_pool = true,
        uint16_t n_streams = 1,
        const std::string& tuning_cache_path = "cache.json")
        : enable_profiling(profiling)
        , meaningful_kernels_names(decorate_kernel_names)
        , dump_custom_program(dump_custom_program)
        , compiler_options(options)
        , single_kernel_name(single_kernel)
        , enable_parallelisation(primitives_parallelisation)
        , engine_log(engine_log)
        , sources_dumps_dir(sources_dumps_dir)
        , priority_mode(priority_mode)
        , throttle_mode(throttle_mode)
        , enable_memory_pool(memory_pool)
        , n_streams(n_streams)
        , tuning_cache_path(tuning_cache_path) {
        if (n_streams == 0) {
            throw std::invalid_argument("Invalid streams count set in engine config");
        }
    }
};

struct engine_impl;

/// @brief Represents clDNN engine object.
struct engine {
    /// @brief Constructs @p OpenCL engine
    explicit engine(const engine_configuration& configuration = engine_configuration())
        : engine(engine_types::ocl, device::create_default(), configuration) {}

    /// @brief Constructs @p OpenCL engine
    explicit engine(const device& device, const engine_configuration& configuration = engine_configuration())
        : engine(engine_types::ocl, device, configuration) {}

    /// @brief Construct engine of the specified @p type, @p engine_num, and @p configuration options.
    /// @param[in] type Engine type @ref cldnn_engine_type. Only OCL engine is supported.
    /// @param[in] engine_num Engine index. Should be 0.
    /// @param[in] configuration Engine configuration options.
    engine(engine_types type, const device& device, const engine_configuration& configuration = engine_configuration());

    // TODO add move construction/assignment
    engine(const engine& other) : _impl(other._impl) {
        retain();
    }

    engine& operator=(const engine& other) {
        if (_impl == other._impl) return *this;
        release();
        _impl = other._impl;
        retain();
        return *this;
    }

    ~engine() {
        release();
    }

    friend bool operator==(const engine& lhs, const engine& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const engine& lhs, const engine& rhs) { return !(lhs == rhs); }

    /// @brief Returns number of available engines of the particular @p type.
    static uint32_t engine_count(engine_types type);

    /// @brief Release pending memory allocated in OpenCL context.
    void release_pending_memory(uint32_t net_id) const;

    /// @brief Returns information about properties and capabilities of the device used for allocation of the engine.
    device_info get_info() const;

    /// @brief Returns OpenCL context handle of the engine.
    void* get_context() const;

    /// @brief Returns total size of all resources allocated using given engine
    uint64_t get_max_used_device_memory_size() const;

    /// @brief Returns total size of currently resources allocated using given engine
    uint64_t get_temp_used_device_memory_size() const;

    /// @brief Returns type of the engine.
    engine_types get_type() const;

    /// @brief get C API engine handler.
    engine_impl* get() const { return _impl; }

private:
    friend struct network;
    friend struct memory;
    friend struct event;

    engine_impl* _impl;

    void retain();
    void release();
};
CLDNN_API_CLASS(engine)

/// @}

/// @}

}  // namespace cldnn
