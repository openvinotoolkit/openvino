// Copyright (C) 2018-2025 Intel Corporation
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

#    define ADD_MATCHER(obj, region, ...)            obj->add_matcher<region>(__VA_ARGS__);
#    define REGISTER_PASS(obj, region, ...)          obj.register_pass<region>(__VA_ARGS__);
#    define REGISTER_DISABLED_PASS(obj, region, ...) obj.register_pass<region, false>(__VA_ARGS__);

#    define OV_PASS_CALLBACK(matcher)                                   \
        openvino::itt::handle_t m_callback_handle;                      \
        m_callback_handle = openvino::itt::handle(matcher->get_name()); \
        OV_ITT_SCOPED_TASK(SIMPLE_ov_pass, m_callback_handle)
#elif defined(SELECTIVE_BUILD)

#    define MATCHER_SCOPE_(scope, region)                              \
        if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(scope, _, region)) == 0) \
        OPENVINO_THROW(std::string(OV_PP_TOSTRING(OV_PP_CAT3(scope, _, region))) + " is disabled!")

#    define MATCHER_SCOPE(region)                                        \
        const std::string matcher_name(OV_PP_TOSTRING(region));          \
        if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, region)) == 0) \
        return
#    define RUN_ON_FUNCTION_SCOPE(region) MATCHER_SCOPE_(ov_pass, OV_PP_CAT(region, _run_on_function))

#    define RUN_ON_MODEL_SCOPE(region) MATCHER_SCOPE_(ov_pass, OV_PP_CAT(region, _run_on_model))

#    define ADD_MATCHER_OBJ_1(obj, region, ...) obj->add_matcher<region>(__VA_ARGS__);
#    define ADD_MATCHER_OBJ_0(obj, region, ...)
#    define ADD_MATCHER(obj, region, ...)                                                   \
        OV_PP_CAT(ADD_MATCHER_OBJ_, OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, region))) \
        (obj, region, __VA_ARGS__)

#    define REGISTER_PASS_1(obj, region, ...) obj.register_pass<region>(__VA_ARGS__);
#    define REGISTER_PASS_0(obj, region, ...)
#    define PASS_DEFAULT(region)      OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, region))
#    define PASS_RUN_ON_MODEL(region) OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, OV_PP_CAT(region, _run_on_model)))
#    define PASS_RUN_ON_FUNCTION(region) \
        OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, OV_PP_CAT(region, _run_on_function)))

#    define OV_PP_FIRST_ARG(...)          OV_PP_EXPAND(OV_PP_FIRST_ARG_(__VA_ARGS__, 0))
#    define OV_PP_FIRST_ARG_(...)         OV_PP_EXPAND(OV_PP_FIRST_ARG_GET(__VA_ARGS__))
#    define OV_PP_FIRST_ARG_GET(val, ...) val

#    define OV_OR_ARG_PLACEHOLDER_1 1,
#    define OV_OR_ARG_PLACEHOLDER_0
#    define OV_OR_(arg1_or_junk, arg2_or_junk) OV_PP_FIRST_ARG(arg1_or_junk arg2_or_junk 0)
#    define OV_OR_2(x, y)                      OV_OR_(OV_PP_CAT(OV_OR_ARG_PLACEHOLDER_, x), OV_PP_CAT(OV_OR_ARG_PLACEHOLDER_, y))
#    define OV_OR_3(x, y, z)                   OV_OR_2(OV_OR_2(x, y), z)
#    define REGISTER_PASS(obj, region, ...)                                                               \
        OV_PP_CAT(REGISTER_PASS_,                                                                         \
                  OV_OR_3(PASS_DEFAULT(region), PASS_RUN_ON_MODEL(region), PASS_RUN_ON_FUNCTION(region))) \
        (obj, region, __VA_ARGS__)

#    define REGISTER_PASS_WITH_FALSE_1(obj, ...) obj.register_pass<region, false>(__VA_ARGS__);
#    define REGISTER_PASS_WITH_FALSE_0(obj, ...)
#    define REGISTER_DISABLED_PASS(obj, region, ...)                                                 \
        OV_PP_CAT(REGISTER_PASS_WITH_FALSE_, OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, region))) \
        (obj, region, __VA_ARGS__)
#    define OV_PASS_CALLBACK(matcher)
#else

#    define MATCHER_SCOPE(region) const std::string matcher_name(OV_PP_TOSTRING(region))
#    define RUN_ON_FUNCTION_SCOPE(region)
#    define RUN_ON_MODEL_SCOPE(region)

#    define ADD_MATCHER(obj, region, ...)            obj->add_matcher<region>(__VA_ARGS__);
#    define REGISTER_PASS(obj, region, ...)          obj.register_pass<region>(__VA_ARGS__);
#    define REGISTER_DISABLED_PASS(obj, region, ...) obj.register_pass<region, false>(__VA_ARGS__);
#    define OV_PASS_CALLBACK(matcher)
#endif

#define ADD_MATCHER_FOR_THIS(region, ...) ADD_MATCHER(this, region, __VA_ARGS__)
