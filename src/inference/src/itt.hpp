// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file itt.hpp
 */

#pragma once

#include "openvino/cc/selective_build.h"
#include "openvino/itt.hpp"

namespace ov {
namespace itt {
namespace domains {
OV_ITT_DOMAIN(OV);
OV_ITT_DOMAIN(ReadTime);
OV_ITT_DOMAIN(LoadTime);
}  // namespace domains
}  // namespace itt
}  // namespace ov

OV_CC_DOMAINS(ir_reader);

#if defined(SELECTIVE_BUILD_ANALYZER)

#    define IR_READER_SCOPE(region) OV_SCOPE(ir_reader, region)

#elif defined(SELECTIVE_BUILD)

#    define IR_READER_SCOPE(region)                                        \
        if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ir_reader, _, region)) == 0) \
        OPENVINO_THROW(OV_PP_TOSTRING(OV_PP_CAT3(ir_reader, _, region)), " is disabled!")

#else

#    define IR_READER_SCOPE(region)

#endif
