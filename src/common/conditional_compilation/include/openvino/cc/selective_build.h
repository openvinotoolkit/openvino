// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/*
        This file contains a useful API for analyzing OpenVINO sources
    and enabling or disabling some regions.
        Three working modes are currently supported.
    * SELECTIVE_BUILD_ANALYZER  This macro enables analysis mode for annotated code regions.
    *                           When the process completes, a new C++ header file is created
    *                           that contains macros for enabling active regions. This file
    *                           should be included in all analyzed C++ files.
    *
    * SELECTIVE_BUILD           This mode disables inactive areas of the code using the result
    *                           of the analysis step.
    *
    * No definitions            The default behavior is kept if no SELECTIVE_BUILD* macros are defined,
    *                           i.e all features of the OpenVINO are enabled.
    *
    * Prerequisites:
    *   Before using macros for code annotation, domains for conditional
    * compilation should be defined in module namespace.
    *
    *   OV_CC_DOMAINS(MyModule);
    *
    * An example of using annotation:
    *
    *  I. Any C++ code block:
    *       OV_SCOPE(MyModule, ScopeName) {
    *           // Any C++ code.
    *           cout << "Hello world!";
    *       }
    *
    *  II. Template class instantiation using switch-case:
    *
    *    struct Context { ... };
    *
    *    template<typename T>
    *    struct SomeTemplateClass {
    *        void operator()(Context &context) {
    *           // Any C++ code.
    *           cout << "Hello world!";
    *        }
    *    };
    *
    *    auto key = Precision::U8;
    *    Context context;
    *
    *    OV_SWITCH(MyModule, SomeTemplateClass, context, key,
    *        OV_CASE(Precision::U8, uint8_t),
    *        OV_CASE(Precision::I8, int8_t),
    *        OV_CASE(Precision::FP32, float));
    *
*/

#include <openvino/itt.hpp>
#include <openvino/util/pp.hpp>

#define OV_CC_EXPAND   OV_PP_EXPAND
#define OV_CC_CAT      OV_PP_CAT
#define OV_CC_TOSTRING OV_PP_TOSTRING

#ifdef SELECTIVE_BUILD_ANALYZER
#    include <string>
#endif

#include <tuple>
#include <utility>

namespace openvino {
namespace cc {

#ifndef SELECTIVE_BUILD_ANALYZER

namespace internal {

template <typename C, typename T>
struct case_wrapper {
    using type = T;
    const C value{};

    case_wrapper(C&& val) : value(std::forward<C>(val)) {}
};

template <typename T, typename C>
case_wrapper<C, T> make_case_wrapper(C&& val) {
    return case_wrapper<C, T>(std::forward<C>(val));
}

template <template <typename...> class Fn, typename Ctx, typename T, typename Case>
bool match(Ctx&& ctx, T&& val, Case&& cs) {
    const bool is_matched = val == cs.value;
    if (is_matched)
        Fn<typename Case::type>()(ctx);
    return is_matched;
}

template <template <typename...> class Fn, typename Ctx, typename T, typename Case, typename... Cases>
bool match(Ctx&& ctx, T&& val, Case&& cs, Cases&&... cases) {
    if (match<Fn>(ctx, val, std::forward<Case>(cs)))
        return true;
    return match<Fn>(std::forward<Ctx>(ctx), std::forward<T>(val), std::forward<Cases>(cases)...);
}

}  // namespace internal

#endif  // SELECTIVE_BUILD_ANALYZER

#ifdef SELECTIVE_BUILD_ANALYZER  // OpenVINO analysis

#    define OV_CC_DOMAINS(Module)                                                                      \
        OV_ITT_DOMAIN(OV_PP_CAT(SIMPLE_, Module));  /* Domain for simple scope surrounded by ifdefs */ \
        OV_ITT_DOMAIN(OV_PP_CAT(SWITCH_, Module));  /* Domain for switch/cases */                      \
        OV_ITT_DOMAIN(OV_PP_CAT(FACTORY_, Module)); /* Domain for factories */

namespace internal {

template <typename C, typename T>
struct case_wrapper {
    using type = T;
    const C value{};
    const char* name = nullptr;

    case_wrapper(C&& val, const char* name) : value(std::forward<C>(val)), name(name) {}
};

template <typename T, typename C>
case_wrapper<C, T> make_case_wrapper(C&& val, const char* name) {
    return case_wrapper<C, T>(std::forward<C>(val), name);
}

template <openvino::itt::domain_t (*domain)(), template <typename...> class Fn, typename Ctx, typename T, typename Case>
bool match(char const* region, Ctx&& ctx, T&& val, Case&& cs) {
    const bool is_matched = val == cs.value;
    if (is_matched) {
        openvino::itt::ScopedTask<domain> task(openvino::itt::handle(std::string(region) + "$" + cs.name));
        Fn<typename Case::type>()(std::forward<Ctx>(ctx));
    }
    return is_matched;
}

template <openvino::itt::domain_t (*domain)(),
          template <typename...>
          class Fn,
          typename Ctx,
          typename T,
          typename Case,
          typename... Cases>
bool match(char const* region, Ctx&& ctx, T&& val, Case&& cs, Cases&&... cases) {
    if (match<domain, Fn>(region, std::forward<Ctx>(ctx), std::forward<T>(val), std::forward<Case>(cs)))
        return true;
    return match<domain, Fn>(region, std::forward<Ctx>(ctx), std::forward<T>(val), std::forward<Cases>(cases)...);
}

}  // namespace internal

#    define OV_SCOPE(Module, region) OV_ITT_SCOPED_TASK(OV_PP_CAT(SIMPLE_, Module), OV_PP_TOSTRING(region));

#    define OV_SWITCH(Module, fn, ctx, val, ...) \
        openvino::cc::internal::match<OV_PP_CAT(SWITCH_, Module), fn>(OV_PP_TOSTRING(fn), ctx, val, __VA_ARGS__);

#    define OV_CC_LBR (
#    define OV_CC_RBR )

#    define OV_CASE(Case, Type) \
        openvino::cc::internal::make_case_wrapper<Type>(Case, OV_PP_TOSTRING(OV_CASE OV_CC_LBR Case, Type OV_CC_RBR))

#    define OV_CASE2(Case1, Case2, Type1, Type2)                             \
        openvino::cc::internal::make_case_wrapper<std::tuple<Type1, Type2>>( \
            std::make_tuple(Case1, Case2),                                   \
            OV_PP_TOSTRING(OV_CASE2 OV_CC_LBR Case1, Case2, Type1, Type2 OV_CC_RBR))

#elif defined(SELECTIVE_BUILD)  // OpenVINO selective build is enabled

#    define OV_CC_DOMAINS(Module)

#    define OV_CC_SCOPE_IS_ENABLED OV_PP_IS_ENABLED

#    define OV_SCOPE(Module, region)                                                                        \
        for (bool ovCCScopeIsEnabled = OV_PP_IS_ENABLED(OV_PP_CAT3(Module, _, region)); ovCCScopeIsEnabled; \
             ovCCScopeIsEnabled = false)

// Switch is disabled
#    define OV_CC_SWITCH_0(Module, fn, ctx, val)

// Switch is enabled
#    define OV_CC_SWITCH_1(Module, fn, ctx, val) \
        openvino::cc::internal::match<fn>(ctx, val, OV_PP_CAT4(Module, _, fn, _cases));

#    define OV_SWITCH(Module, fn, ctx, val, ...) \
        OV_PP_EXPAND(OV_PP_CAT(OV_CC_SWITCH_, OV_PP_IS_ENABLED(OV_PP_CAT3(Module, _, fn)))(Module, fn, ctx, val))

#    define OV_CASE(Case, Type) openvino::cc::internal::make_case_wrapper<Type>(Case)

#    define OV_CASE2(Case1, Case2, Type1, Type2) \
        openvino::cc::internal::make_case_wrapper<std::tuple<Type1, Type2>>(std::make_tuple(Case1, Case2))

#else

#    define OV_CC_DOMAINS(Module)

#    define OV_SCOPE(Module, region)

#    define OV_SWITCH(Module, fn, ctx, val, ...) openvino::cc::internal::match<fn>(ctx, val, __VA_ARGS__);

#    define OV_CASE(Case, Type) openvino::cc::internal::make_case_wrapper<Type>(Case)

#    define OV_CASE2(Case1, Case2, Type1, Type2) \
        openvino::cc::internal::make_case_wrapper<std::tuple<Type1, Type2>>(std::make_tuple(Case1, Case2))

#endif

}  // namespace cc
}  // namespace openvino
