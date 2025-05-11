// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file itt.h
 */

#pragma once

#include <openvino/cc/factory.h>
#include <openvino/cc/selective_build.h>

#include <openvino/itt.hpp>

namespace ov {
namespace intel_cpu {
namespace itt {
namespace domains {
OV_ITT_DOMAIN(intel_cpu);
OV_ITT_DOMAIN(intel_cpu_LT);
}  // namespace domains
}  // namespace itt
}  // namespace intel_cpu
}  // namespace ov

#if defined(SELECTIVE_BUILD_ANALYZER)
#    define CPU_LPT_SCOPE(region)             OV_SCOPE(intel_cpu, region)
#    define CPU_GRAPH_OPTIMIZER_SCOPE(region) OV_SCOPE(intel_cpu, region)
#elif defined(SELECTIVE_BUILD)
#    define CPU_LPT_SCOPE(region)                                          \
        if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(intel_cpu, _, region)) == 0) \
        OPENVINO_THROW(std::string(OV_PP_TOSTRING(OV_PP_CAT3(ov_op, _, region))), " is disabled!")
#    define CPU_GRAPH_OPTIMIZER_SCOPE(region)                              \
        if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(intel_cpu, _, region)) == 0) \
        OPENVINO_THROW(std::string(OV_PP_TOSTRING(OV_PP_CAT3(intel_cpu, _, region))), " is disabled!")
#else
#    define CPU_LPT_SCOPE(region) OV_ITT_SCOPED_TASK(ov::intel_cpu::itt::domains::intel_cpu, OV_PP_TOSTRING(region))
#    define CPU_GRAPH_OPTIMIZER_SCOPE(region)
#endif
