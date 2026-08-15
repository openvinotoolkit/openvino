// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Core-local execution configuration. In the standalone core this is a plain
// value holder: the ov::PluginConfig / ov::AnyMap / ov::Model based property
// machinery of the upstream plugin is intentionally dropped. Only the getters
// actually consumed by the runtime are kept, with upstream-default values.

#pragma once

#include <string>

#include "internal_properties.hpp"

namespace ov::intel_gpu {

struct ExecutionConfig {
    ExecutionConfig() = default;
    ExecutionConfig(const ExecutionConfig&) = default;
    ExecutionConfig& operator=(const ExecutionConfig&) = default;

    // --- static flags (upstream GPU plugin defaults) ---
    static bool get_disable_usm() { return false; }
    static bool get_dump_profiling_data_per_iter() { return false; }
    static int get_verbose() { return 0; }
    static bool get_verbose_color() { return false; }
    static const std::string& get_log_to_file() {
        static const std::string empty;
        return empty;
    }

    // --- instance getters used by the runtime ---
    bool get_enable_profiling() const { return m_enable_profiling; }
    QueueTypes get_queue_type() const { return m_queue_type; }
    bool get_disable_memory_reuse() const { return m_disable_memory_reuse; }
    bool get_dump_memory_pool() const { return m_dump_memory_pool; }
    float get_mem_pool_util_threshold() const { return m_mem_pool_util_threshold; }
    const std::string& get_dump_profiling_data_path() const { return m_dump_profiling_data_path; }
    const std::string& get_average_counters() const { return m_average_counters; }

    ExecutionConfig clone() const { return *this; }

private:
    bool m_enable_profiling = false;
    QueueTypes m_queue_type = QueueTypes::in_order;
    bool m_disable_memory_reuse = false;
    bool m_dump_memory_pool = false;
    float m_mem_pool_util_threshold = 0.0f;
    std::string m_dump_profiling_data_path;
    std::string m_average_counters;
};

}  // namespace ov::intel_gpu
