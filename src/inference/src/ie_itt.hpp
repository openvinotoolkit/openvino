// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file ie_itt.hpp
 */

#pragma once

#include <openvino/itt.hpp>
#include <openvino/cc/selective_build.h>

namespace InferenceEngine {
namespace itt {
namespace domains {
OV_ITT_DOMAIN(IE_LT);
OV_ITT_DOMAIN(IE_CORE);
}  // namespace domains
}  // namespace itt
}  // namespace InferenceEngine
OV_CC_DOMAINS(IE_CORE);

namespace ov {
namespace itt {
namespace domains {
OV_ITT_DOMAIN(IE);
OV_ITT_DOMAIN(IE_RT);
}  // namespace domains
}  // namespace itt
}  // namespace ov

#if defined(SELECTIVE_BUILD_ANALYZER)
#    define OV_CORE_SCOPE(region) OV_SCOPE(IE_CORE, region)
#elif defined(SELECTIVE_BUILD)
#    define OV_CORE_SCOPE(region)                                        \
        if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(IE_CORE, _, region)) == 0) \
        throw ngraph::ngraph_error(std::string(OV_PP_TOSTRING(OV_PP_CAT3(IE_CORE, _, region))) + " is disabled!")
#else
#    define OV_CORE_SCOPE(region)
#endif
