// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino tbbbind domains for tracing
 * @file openvino/runtime/threading/itt.hpp
 */

#pragma once

#include "openvino/cc/selective_build.h"
#include "openvino/core/except.hpp"
#include "openvino/itt.hpp"

namespace ov {
namespace tbbbind {
namespace itt {
namespace domains {
OV_ITT_DOMAIN(tbb_bind);
}  // namespace domains
}  // namespace itt
}  // namespace tbbbind
}  // namespace ov

OV_CC_DOMAINS(tbb_bind);

/*
 * TBB_BIND_SCOPE macro allows to disable parts of tbb_bind calling if they are not used.
 */
#if defined(SELECTIVE_BUILD_ANALYZER)

#    define TBB_BIND_SCOPE(region) OV_SCOPE(tbb_bind, region)
#    define TBB_BIND_NUMA_ENABLED  OV_SCOPE(tbb_bind, NUMA)

#elif defined(SELECTIVE_BUILD)

#    define TBB_BIND_SCOPE(region)                                        \
        if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(tbb_bind, _, NUMA)) == 1 && \
            OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(tbb_bind, _, region)) == 1)
#    define TBB_BIND_NUMA_ENABLED

#else

#    define TBB_BIND_SCOPE(region)
#    define TBB_BIND_NUMA_ENABLED
#endif

