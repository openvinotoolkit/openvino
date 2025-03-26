// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cstring>
#include <iostream>
#include <filesystem>

#include "intel_gpu/runtime/execution_config.hpp"
namespace ov::intel_gpu {

// Verbose log levels:
// DISABLED - silent mode (Default)
// INFO - Minimal verbose:
//     * May log basic info about device, plugin configuration, model and execution statistics
//     * Mustn't log any info that depend on neither number of iterations or number of layers in the model
//     * Minimal impact on both load time and inference time
// LOG - Enables graph optimization verbose:
//     * Includes info from log level INFO
//     * May log info about applied graph transformations, memory allocations and other model compilation time steps
//     * May impact compile_model() execution time
//     * Minimal impact on inference time
// TRACE - Enables basic execution time verbose
//     * Includes info from log level LOG
//     * May log info during model execution
//     * May log short info about primitive execution
//     * May impact network execution time
// TRACE_DETAIL - Max verbosity
//     * Includes info from log level TRACE
//     * May log any stage and print detailed info about each execution step
enum class LogLevel : int8_t {
    DISABLED = 0,
    INFO = 1,
    LOG = 2,
    TRACE = 3,
    TRACE_DETAIL = 4
};

std::ostream& get_verbose_stream();
}  // namespace ov::intel_gpu

#ifdef GPU_DEBUG_CONFIG

namespace color {
static constexpr const char* dark_gray = "\033[1;30m";
static constexpr const char* blue      = "\033[1;34m";
static constexpr const char* purple    = "\033[1;35m";
static constexpr const char* cyan      = "\033[1;36m";
static constexpr const char* reset     = "\033[0m";
}  // namespace color

static constexpr const char* prefix = "GPU_Debug: ";

#define GPU_DEBUG_IF(cond) if (cond)
#define GPU_DEBUG_VALUE_OR(debug_value, release_value) debug_value
#define GPU_DEBUG_CODE(...) __VA_ARGS__

#define GPU_DEBUG_DEFINE_MEM_LOGGER(stage) \
    cldnn::instrumentation::mem_usage_logger mem_logger{stage, ov::intel_gpu::ExecutionConfig::get_verbose() >= 2};

#define GPU_DEBUG_PROFILED_STAGE(stage)                                       \
    auto stage_prof = cldnn::instrumentation::profiled_stage<primitive_inst>( \
        !get_config().get_dump_profiling_data_path().empty(), *this, stage)

#define GPU_DEBUG_PROFILED_STAGE_CACHE_HIT(val) stage_prof.set_cache_hit(val)
#define GPU_DEBUG_PROFILED_STAGE_MEMALLOC_INFO(info) stage_prof.add_memalloc_info(info)

#define GPU_DEBUG_LOG_PREFIX ov::intel_gpu::get_verbose_stream() \
                             << prefix \
                             << std::filesystem::path(__FILE__).filename().generic_string() << ":" \
                             << std::to_string(__LINE__) << ":" \
                             << __func__ << ": "

#define GPU_DEBUG_LOG_COLOR_PREFIX ov::intel_gpu::get_verbose_stream() \
                                   << color::dark_gray << std::string(prefix) \
                                   << color::blue << std::filesystem::path(__FILE__).filename().generic_string() << ":" \
                                   << color::purple << std::to_string(__LINE__) << ":" \
                                   << color::cyan << __func__ << ": " << color::reset

#define GPU_DEBUG_LOG_RAW_INT(min_verbose_level) if (ov::intel_gpu::ExecutionConfig::get_verbose() >= min_verbose_level) \
    (ov::intel_gpu::ExecutionConfig::get_verbose_color() ? GPU_DEBUG_LOG_COLOR_PREFIX : GPU_DEBUG_LOG_PREFIX)

#define GPU_DEBUG_LOG_RAW(min_verbose_level) \
    GPU_DEBUG_LOG_RAW_INT(static_cast<std::underlying_type_t<ov::intel_gpu::LogLevel>>(min_verbose_level))
#else
#define GPU_DEBUG_IF(cond) if (0)
#define GPU_DEBUG_VALUE_OR(debug_value, release_value) release_value
#define GPU_DEBUG_CODE(...)
#define GPU_DEBUG_DEFINE_MEM_LOGGER(stage)
#define GPU_DEBUG_PROFILED_STAGE(stage)
#define GPU_DEBUG_PROFILED_STAGE_CACHE_HIT(val)
#define GPU_DEBUG_PROFILED_STAGE_MEMALLOC_INFO(info)
#define GPU_DEBUG_LOG_RAW(min_verbose_level) if (0) ov::intel_gpu::get_verbose_stream()
#endif

#define GPU_DEBUG_COUT              GPU_DEBUG_LOG_RAW(ov::intel_gpu::LogLevel::DISABLED)
#define GPU_DEBUG_INFO              GPU_DEBUG_LOG_RAW(ov::intel_gpu::LogLevel::INFO)
#define GPU_DEBUG_LOG               GPU_DEBUG_LOG_RAW(ov::intel_gpu::LogLevel::LOG)
#define GPU_DEBUG_TRACE             GPU_DEBUG_LOG_RAW(ov::intel_gpu::LogLevel::TRACE)
#define GPU_DEBUG_TRACE_DETAIL      GPU_DEBUG_LOG_RAW(ov::intel_gpu::LogLevel::TRACE_DETAIL)
