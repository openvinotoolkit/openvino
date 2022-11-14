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

#    define ADD_MATCHER_FOR_THIS(nspace, region, ...)        add_matcher<nspace::region>(__VA_ARGS__);
#    define ADD_MATCHER_FOR_THIS_WITHOUT_NSPACE(region, ...) add_matcher<region>(__VA_ARGS__);
#    define ADD_MATCHER(obj, nspace, region, ...)            obj->add_matcher<nspace::region>(__VA_ARGS__);
#    define ADD_MATCHER_WITHOUT_NSPACE(obj, region, ...)     obj->add_matcher<region>(__VA_ARGS__);
#    define REGISTER_PASS(obj, nspace, region, flag, ...)    obj.register_pass<nspace::region>(__VA_ARGS__);
#    define REGISTER_DISABLED_PASS(obj, nspace, region, ...) obj.register_pass<nspace::region, false>(__VA_ARGS__);
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

#    define ADD_MATCHER_NO_OBJ_1(region, ...) add_matcher<region>(__VA_ARGS__);
#    define ADD_MATCHER_NO_OBJ_0(region, ...)
#    define ADD_MATCHER_FOR_THIS(nspace, region, ...)                                          \
        OV_PP_CAT(ADD_MATCHER_NO_OBJ_, OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, region))) \
        (nspace::region, __VA_ARGS__)
#    define ADD_MATCHER_FOR_THIS_WITHOUT_NSPACE(region, ...)                                   \
        OV_PP_CAT(ADD_MATCHER_NO_OBJ_, OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, region))) \
        (region, __VA_ARGS__)
#    define ADD_MATCHER_OBJ_1(obj, region, ...) obj->add_matcher<region>(__VA_ARGS__);
#    define ADD_MATCHER_OBJ_0(obj, region, ...)
#    define ADD_MATCHER(obj, nspace, region, ...)                                           \
        OV_PP_CAT(ADD_MATCHER_OBJ_, OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, region))) \
        (obj, nspace::region, __VA_ARGS__)
#    define ADD_MATCHER_WITHOUT_NSPACE(obj, region, ...)                                    \
        OV_PP_CAT(ADD_MATCHER_OBJ_, OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, region))) \
        (obj, region, __VA_ARGS__)
#    define REGISTER_PASS_1(obj, region, ...) obj.register_pass<region>(__VA_ARGS__);
#    define REGISTER_PASS_0(obj, region, ...)
#    define REGISTER_PASS(obj, nspace, region, flag, ...)                                                  \
        OV_PP_CAT(REGISTER_PASS_, OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, OV_PP_CAT(region, flag)))) \
        (obj, nspace::region, __VA_ARGS__)
#    define REGISTER_PASS_WITH_FALSE_1(obj, region, ...) obj.register_pass<region, false>(__VA_ARGS__);
#    define REGISTER_PASS_WITH_FALSE_0(obj, region, ...)
#    define REGISTER_DISABLED_PASS(obj, nspace, region, ...)                                         \
        OV_PP_CAT(REGISTER_PASS_WITH_FALSE_, OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, region))) \
        (obj, nspace::region, __VA_ARGS__)
#else

#    define MATCHER_SCOPE(region) const std::string matcher_name(OV_PP_TOSTRING(region))
#    define RUN_ON_FUNCTION_SCOPE(region)
#    define RUN_ON_MODEL_SCOPE(region)

#    define ADD_MATCHER_FOR_THIS(nspace, region, ...)        add_matcher<nspace::region>(__VA_ARGS__);
#    define ADD_MATCHER_FOR_THIS_WITHOUT_NSPACE(region, ...) add_matcher<region>(__VA_ARGS__);
#    define ADD_MATCHER(obj, nspace, region, ...)            obj->add_matcher<nspace::region>(__VA_ARGS__);
#    define ADD_MATCHER_WITHOUT_NSPACE(obj, region, ...)     obj->add_matcher<region>(__VA_ARGS__);
#    define REGISTER_PASS(obj, nspace, region, flag, ...)    obj.register_pass<nspace::region>(__VA_ARGS__);
#    define REGISTER_DISABLED_PASS(obj, nspace, region, ...) obj.register_pass<nspace::region, false>(__VA_ARGS__);
#endif
