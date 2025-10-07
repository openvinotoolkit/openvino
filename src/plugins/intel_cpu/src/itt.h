// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file itt.h
 */

#pragma once

#if defined(SELECTIVE_BUILD_ANALYZER)
#    include "openvino/cc/selective_build.h"
#elif defined(SELECTIVE_BUILD)
#    include "openvino/util/pp.hpp"
#endif

#include <openvino/itt.hpp>
#include <string>

#include "../../core/src/itt.hpp"

namespace ov::intel_cpu::itt::domains {
OV_ITT_DOMAIN(ov_intel_cpu, "ov::intel_cpu");
OV_ITT_DOMAIN(ov_intel_cpu_LT, "ov::intel_cpu::lt");
}  // namespace ov::intel_cpu::itt::domains

namespace ov::intel_cpu::itt {

class ScopedOpExecTask {
public:
    explicit ScopedOpExecTask(const char* name) noexcept : ScopedOpExecTask(openvino::itt::handle(name)) {}
    explicit ScopedOpExecTask(const std::string& name) noexcept : ScopedOpExecTask(name.c_str()) {}
    explicit ScopedOpExecTask(openvino::itt::handle_t handle) noexcept : m_handle(handle) {
        openvino::itt::internal::taskBegin(::ov::itt::domains::ov_op_exec(), m_handle);
    }
    ~ScopedOpExecTask() noexcept {
        openvino::itt::internal::taskEnd(::ov::itt::domains::ov_op_exec());
    }

    ScopedOpExecTask(const ScopedOpExecTask&) = delete;
    ScopedOpExecTask& operator=(const ScopedOpExecTask&) = delete;

private:
    openvino::itt::handle_t m_handle{};
};

}  // namespace ov::intel_cpu::itt

#define OV_CPU_NODE_SCOPE_CONCAT_IMPL(x, y) x##y
#define OV_CPU_NODE_SCOPE_CONCAT(x, y)      OV_CPU_NODE_SCOPE_CONCAT_IMPL(x, y)
#define OV_CPU_NODE_SCOPED_TASK(taskName) \
    ::ov::intel_cpu::itt::ScopedOpExecTask OV_CPU_NODE_SCOPE_CONCAT(cpuNodeScopedTaskGuard, __LINE__)(taskName)

#if defined(SELECTIVE_BUILD_ANALYZER)
#    define CPU_LPT_SCOPE(region)             OV_SCOPE(intel_cpu, region)
#    define CPU_GRAPH_OPTIMIZER_SCOPE(region) OV_SCOPE(intel_cpu, region)
#elif defined(SELECTIVE_BUILD)
#    define CPU_LPT_SCOPE(region)                                          \
        if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(intel_cpu, _, region)) == 0) \
        OPENVINO_THROW(std::string(OV_PP_TOSTRING(OV_PP_CAT3(intel_cpu, _, region))), " is disabled!")
#    define CPU_GRAPH_OPTIMIZER_SCOPE(region)                              \
        if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(intel_cpu, _, region)) == 0) \
        OPENVINO_THROW(std::string(OV_PP_TOSTRING(OV_PP_CAT3(intel_cpu, _, region))), " is disabled!")
#else
#    define CPU_LPT_SCOPE(region) OV_ITT_SCOPED_TASK(ov::intel_cpu::itt::domains::ov_intel_cpu, OV_PP_TOSTRING(region))
#    define CPU_GRAPH_OPTIMIZER_SCOPE(region)
#endif
