// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/cc/selective_build.h>

#include <openvino/itt.hpp>

OV_CC_DOMAINS(ov_pass);

/*
 * RUN_ON_MODEL_SCOPE macro allows to disable the run_on_function pass
 * RUN_ON_FUNCTION_SCOPE macro allows to disable the run_on_function pass
 * MATCHER_SCOPE macro allows to disable the MatcherPass if matcher isn't applied
 */
#if defined(SELECTIVE_BUILD_ANALYZER)

#    define RUN_ON_FUNCTION_SCOPE(region) OV_SCOPE(ov_pass, OV_PP_CAT(region, _run_on_function))
#    define MATCHER_SCOPE(region)         const std::string matcher_name(OV_PP_TOSTRING(region))
#    define RUN_ON_MODEL_SCOPE(region)    OV_SCOPE(ov_pass, OV_PP_CAT(region, _run_on_model))
#    define MATCHER_SCOPE_ENABLE(region)  OV_SCOPE(ov_pass, region)

#elif defined(SELECTIVE_BUILD)

#    define MATCHER_SCOPE_(scope, region)                              \
        if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(scope, _, region)) == 0) \
        throw ngraph::ngraph_error(std::string(OV_PP_TOSTRING(OV_PP_CAT3(scope, _, region))) + " is disabled!")

#    define MATCHER_SCOPE(region)                                        \
        const std::string matcher_name(OV_PP_TOSTRING(region));          \
        if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, region)) == 0) \
        return
#    define RUN_ON_FUNCTION_SCOPE(region) MATCHER_SCOPE_(ov_pass, OV_PP_CAT(region, _run_on_function))

#    define RUN_ON_MODEL_SCOPE(region) MATCHER_SCOPE_(ov_pass, OV_PP_CAT(region, _run_on_model))

#    define MATCHER_SCOPE_ENABLE(region)
#else

#    define MATCHER_SCOPE(region) const std::string matcher_name(OV_PP_TOSTRING(region))
#    define RUN_ON_FUNCTION_SCOPE(region)
#    define RUN_ON_MODEL_SCOPE(region)
#    define MATCHER_SCOPE_ENABLE(region)
#endif
