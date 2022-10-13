// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/cc/selective_build.h>

#include <openvino/itt.hpp>

OV_CC_DOMAINS(ov_pass);

/*
 * RUN_ON_MODEL_SCOPE macro allows to disable the run_on_model pass
 * RUN_ON_FUNCTION_SCOPE macro allows to disable the run_on_function pass
 * MATCHER_SCOPE macro allows to disable the MatcherPass if matcher isn't applied
 */
#if defined(SELECTIVE_BUILD_ANALYZER)

#    define RUN_ON_FUNCTION_SCOPE(region) OV_SCOPE(ov_pass, OV_PP_CAT(region, _run_on_function))
#    define MATCHER_SCOPE(region)         const std::string matcher_name(OV_PP_TOSTRING(region))
#    define RUN_ON_MODEL_SCOPE(region)    OV_SCOPE(ov_pass, OV_PP_CAT(region, _run_on_model))

#    define ADD_MATCHER_SCOPE_WITHOUT_OBJ(nspace, region, ...)   add_matcher<nspace::region>(__VA_ARGS__);
#    define ADD_MATCHER_SCOPE_WITHOUT_OBJ_NSPACE(region, ...)    add_matcher<region>(__VA_ARGS__);
#    define ADD_MATCHER_SCOPE_WITH_OBJ(obj, nspace, region, ...) obj->add_matcher<nspace::region>(__VA_ARGS__);

#    define CC_TRANSFORMATIONS_MATCH_SCOPE(region)
#    define CC_TRANSFORMATIONS_MODEL_SCOPE(region)
#    define CC_TRANSFORMATIONS_FUNCTION_SCOPE(region)
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

#    define ADD_MATCHER_1(nspace, region, ...) \
        add_matcher<nspace::region>(__VA_ARGS__);           \
        std::cout << #nspace<<"::"<<#region<<" ADD_MATCHER_1\n";
#    define ADD_MATCHER_0(nspace, region, ...) std::cout << #nspace<<"::"<<#region<<" ADD_MATCHER_0\n";

#    define ADD_MATCHER_NO_NSPACE_1(region, ...) \
        add_matcher<region>(__VA_ARGS__);        \
        std::cout << #region << " ADD_MATCHER_NO_NSPACE_1\n";
#    define ADD_MATCHER_NO_NSPACE_0(region, ...) std::cout << #region << " ADD_MATCHER_NO_NSPACE_0\n";

#    define ADD_MATCHER_OBJ_1(obj, nspace, region, ...) \
        obj->add_matcher<nspace::region>(__VA_ARGS__);               \
        std::cout << #obj<<" "<< #nspace<<"::"<<#region << " ADD_MATCHER_OBJ_1\n";
#    define ADD_MATCHER_OBJ_0(obj, nspace, region, ...) std::cout <<#obj<<" "<< #nspace<<"::"<<#region<< " ADD_MATCHER_OBJ_0\n";

#    define ADD_MATCHER_SCOPE_WITHOUT_OBJ(nspace, region, ...)                          \
        OV_PP_CAT(ADD_MATCHER_, OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, region))) \
        (nspace, region)
#    define ADD_MATCHER_SCOPE_WITHOUT_OBJ_NSPACE(region, ...)                                     \
        OV_PP_CAT(ADD_MATCHER_NO_NSPACE_, OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, region))) \
        (region)
#    define ADD_MATCHER_SCOPE_WITH_OBJ(obj, nspace, region, ...)                            \
        OV_PP_CAT(ADD_MATCHER_OBJ_, OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, region))) \
        (obj, nspace, region, __VA_ARGS__)

// #    define MATCHER_SCOPE_WITHOUT_OBJ(nspace, region) \
//         OV_PP_CAT(CALL_HELPER_, OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, region)))(nspace, region, add_matcher)
// #    define MATCHER_SCOPE_WITH_OBJ(obj, nspace, region) \
//         OV_PP_CAT(CALL_HELPER_OBJ_, OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, region)))(obj, . , nspace, region, add_matcher)

// #    define ADD_MATCHER_SCOPE_FUNC_WITHOUT_OBJ(nspace, region, func)                                                  \
//         OV_PP_CAT(CALL_HELPER_, OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, OV_PP_CAT(region, func)))) \
//         (nspace, region, register_pass)
// #    define ADD_MATCHER_SCOPE_FUNC_WITH_OBJ(obj, op, nspace, region, func, ...)                                                    \
//         OV_PP_CAT(CALL_HELPER_OBJ_, OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, OV_PP_CAT(region, func)))) \
//         (obj, op, nspace, region, register_pass, __VA_ARGS__)

#    define CC_TRANSFORMATIONS_MATCH_SCOPE(region) if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, region)) == 1)

// #    define ADD_MATCHER_SCOPE(region) if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, region)) == 1)
#    define CC_TRANSFORMATIONS_MODEL_SCOPE(region) \
        if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, OV_PP_CAT(region, _run_on_model))) == 1)
#    define CC_TRANSFORMATIONS_FUNCTION_SCOPE(region)                                                 \
        if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, OV_PP_CAT(region, _run_on_function))) == 1)
#else

#    define MATCHER_SCOPE(region) const std::string matcher_name(OV_PP_TOSTRING(region))
#    define RUN_ON_FUNCTION_SCOPE(region)
#    define RUN_ON_MODEL_SCOPE(region)

#    define ADD_MATCHER_SCOPE_WITHOUT_OBJ(nspace, region, ...)   add_matcher<nspace::region>(__VA_ARGS__);
#    define ADD_MATCHER_SCOPE_WITHOUT_OBJ_NSPACE(region, ...)    add_matcher<region>(__VA_ARGS__);
#    define ADD_MATCHER_SCOPE_WITH_OBJ(obj, nspace, region, ...) obj->add_matcher<nspace::region>(__VA_ARGS__);

#    define CC_TRANSFORMATIONS_MATCH_SCOPE(region)
#    define CC_TRANSFORMATIONS_MODEL_SCOPE(region)
#    define CC_TRANSFORMATIONS_FUNCTION_SCOPE(region)
#endif
