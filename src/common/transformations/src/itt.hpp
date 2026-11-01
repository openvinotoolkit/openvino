// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file itt.hpp
 */

#pragma once

#include "openvino/cc/selective_build.h"
#include "openvino/itt.hpp"
#include "openvino/cc/pass/itt.hpp"

namespace ov {
namespace pass {
namespace itt {
namespace domains {
    OV_ITT_DOMAIN(Transform, "ov::transformations");
}   // namespace domains
}   // namespace itt
}   // namespace pass
}   // namespace ov

OV_CC_DOMAINS(internal_op);

/*
 * RUN_ON_FUNCTION_SCOPE macro allows to disable the run_on_function pass
 * MATCHER_SCOPE macro allows to disable the MatcherPass if matcher isn't applied
 * INTERNAL_OP_SCOPE macro allows to disable parts of internal openvino operations if they are not used
 * ADD_FUSE_ENTRY macro allows to skip a ConvertPrecision type_to_fuse_map entry if it wasn't used
 */
#if defined(SELECTIVE_BUILD_ANALYZER)

#define INTERNAL_OP_SCOPE(region) OV_SCOPE(internal_op, region)
#define ADD_FUSE_ENTRY_FOR_TYPE_INFO(obj, region, type_info, function)    \
    obj.emplace(type_info, [](const auto& node, const auto& precisions) { \
        OV_SCOPE(ov_pass, OV_PP_CAT(fuse_type_to_, region));              \
        return function(node, precisions);                                \
    })

#elif defined(SELECTIVE_BUILD)

#define INTERNAL_OP_SCOPE(region) MATCHER_SCOPE_(internal_op, region)
#define ADD_FUSE_ENTRY_FOR_TYPE_INFO(obj, region, type_info, function)                \
    do {                                                                              \
        if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _fuse_type_to_, region)) == 1) \
            obj.emplace(type_info, function);                                         \
    } while (0)

#else

#define INTERNAL_OP_SCOPE(region)
#define ADD_FUSE_ENTRY_FOR_TYPE_INFO(obj, region, type_info, function) obj.emplace(type_info, function)

#endif

#define ADD_FUSE_ENTRY(obj, region, op_type, function) \
    ADD_FUSE_ENTRY_FOR_TYPE_INFO(obj, region, op_type::get_type_info_static(), function)
