// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file itt.hpp
 */

#pragma once

#include <openvino/cc/pass/itt.hpp>

namespace ov {
namespace intel_cpu {
namespace itt {
namespace domains {
    OV_ITT_DOMAIN(CPUSpecificTransform);
}   // namespace domains
}   // namespace itt
}   // namespace intel_cpu
}   // namespace ov

OV_CC_DOMAINS(internal_op);

/*
 * RUN_ON_FUNCTION_SCOPE macro allows to disable the run_on_function pass
 * MATCHER_SCOPE macro allows to disable the MatcherPass if matcher isn't applied
 * INTERNAL_OP_SCOPE macro allows to disable parts of internal nGraph operations if they are not used
 */
#if defined(SELECTIVE_BUILD_ANALYZER)

#define INTERNAL_OP_SCOPE(region) OV_SCOPE(internal_op, region)

#elif defined(SELECTIVE_BUILD)

#define INTERNAL_OP_SCOPE(region) MATCHER_SCOPE_(internal_op, region)

#else

#define INTERNAL_OP_SCOPE(region)

#endif
