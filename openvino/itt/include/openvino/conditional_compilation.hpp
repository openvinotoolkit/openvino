// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/*
    This file contains a useful API for analyzing OV plugin sources
    and enabling or disabling some regions.
        Three working modes are currently supported.
    * OV_SELECTIVE_BUILD_LOG This macro enables analysis mode for annotated code regions.
    *                        When the process completes, a new C++ header file is created
    *                        that contains macros for enabling active regions. This file
    *                        should be included in all analysed C++ files.
    * OV_SELECTIVE_BUILD     This mode disables inactive areas of the code using the result
    *                        of the analysis step.
    * No definitions         The default behavior is keept if no OV_SELECTIVE_BUILD * macros are defined,
    *                        i.e all features of the OV plugin are enabled.
    *
    * An example of using annotation:
    *
    *  I. Any C++ code block:
    *       OV_SCOPE(ScopeName,
    *           // Any C++ code.
    *           cout << "Hello world!";
    *       );
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
    *    OV_SWITCH(SomeTemplateClass, context, key,
    *        OV_CASE(Precision::U8, uint8_t),
    *        OV_CASE(Precision::I8, int8_t),
    *        OV_CASE(Precision::FP32, float));
    *
*/

#ifdef OV_SELECTIVE_BUILD_LOG
#include "itt.hpp"
#include <string>
#endif

#include <utility>
#include <tuple>

namespace OVConditionalCompilation {

// Macros for names concatenation
#define OV_CAT_(x, y) x ## y
#define OV_CAT(x, y) OV_CAT_(x, y)
#define OV_CAT3_(x, y, z) x ## y ## z
#define OV_CAT3(x, y, z) OV_CAT3_(x, y, z)

// Expand macro argument
#define OV_EXPAND(x) x

// Macros for string conversion
#define OV_TOSTRING(...) OV_TOSTRING_(__VA_ARGS__)
#define OV_TOSTRING_(...) #__VA_ARGS__

#ifndef OV_SELECTIVE_BUILD_LOG

namespace internal {

template<typename C, typename T>
struct case_wrapper {
    using type = T;
    const C value {};

    case_wrapper(C && val)
        : value(std::forward<C>(val))
    {}
};

template<typename T, typename C>
case_wrapper<C, T> make_case_wrapper(C && val) {
    return case_wrapper<C, T>(std::forward<C>(val));
}

template<template<typename...> typename Fn, typename Ctx, typename T, typename Case>
bool match(Ctx && ctx, T && val, Case && cs) {
    const bool is_matched = val == cs.value;
    if (is_matched)
        Fn<typename Case::type>()(std::forward<Ctx>(ctx));
    return is_matched;
}

template<template<typename...> typename Fn, typename Ctx, typename T, typename Case, typename ...Cases>
bool match(Ctx && ctx, T && val, Case && cs, Cases&&... cases) {
    if (match<Fn>(std::forward<Ctx>(ctx), std::forward<T>(val), std::forward<Case>(cs)))
        return true;
    return match<Fn>(std::forward<Ctx>(ctx), std::forward<T>(val), std::forward<Cases>(cases)...);
}

}   // namespace internal

#endif

#ifdef OV_SELECTIVE_BUILD_LOG           // OV analysis

namespace internal {
namespace itt {
namespace domains {
    OV_ITT_DOMAIN(CC0OV); // Domain for simple scope surrounded by ifdefs
    OV_ITT_DOMAIN(CC1OV); // Domain for switch/cases
    OV_ITT_DOMAIN(CC2OV); // Domain for OV plugin factories
}   // namespace domains
}   // namespace itt

template<typename C, typename T>
struct case_wrapper {
    using type = T;
    const C value {};
    const char *name = nullptr;

    case_wrapper(C && val, const char *name)
        : value(std::forward<C>(val))
        , name(name)
    {}
};

template<typename T, typename C>
case_wrapper<C, T> make_case_wrapper(C && val, const char *name) {
    return case_wrapper<C, T>(std::forward<C>(val), name);
}

template<template<typename...> typename Fn, typename Ctx, typename T, typename Case>
bool match(char const *region, Ctx && ctx, T && val, Case && cs) {
    const bool is_matched = val == cs.value;
    if (is_matched) {
        OV_ITT_SCOPED_TASK(OVConditionalCompilation::internal::itt::domains::CC1OV, std::string(region) + "$" + cs.name);
        Fn<typename Case::type>()(std::forward<Ctx>(ctx));
    }
    return is_matched;
}

template<template<typename...> typename Fn, typename Ctx, typename T, typename Case, typename ...Cases>
bool match(char const *region, Ctx && ctx, T && val, Case && cs, Cases&&... cases) {
    if (match<Fn>(region, std::forward<Ctx>(ctx), std::forward<T>(val), std::forward<Case>(cs)))
        return true;
    return match<Fn>(region, std::forward<Ctx>(ctx), std::forward<T>(val), std::forward<Cases>(cases)...);
}

}   // namespace internal

#define OV_SCOPE(region, ...)                                                         \
    OV_ITT_SCOPED_TASK(OVConditionalCompilation::internal::itt::domains::CC0OV, OV_TOSTRING(region)); \
    __VA_ARGS__

#define OV_SWITCH(fn, ctx, val, ...)                                                \
    internal::match<fn>(OV_TOSTRING(fn), ctx, val, __VA_ARGS__);

#define OV_LBR (
#define OV_RBR )

#define OV_CASE(Case, Type)                                                         \
    internal::make_case_wrapper<Type>(Case, OV_TOSTRING(OV_CASE OV_LBR Case, Type OV_RBR))

#define OV_CASE2(Case1, Case2, Type1, Type2)                                        \
    internal::make_case_wrapper<std::tuple<Type1, Type2>>(                              \
        std::make_tuple(Case1, Case2),                                                  \
        OV_TOSTRING(OV_CASE2 OV_LBR Case1, Case2, Type1, Type2 OV_RBR))

#elif defined(OV_SELECTIVE_BUILD)        // OV subset is used

// Placeholder for first macro argument
#define OV_SCOPE_ARG_PLACEHOLDER_1 0,

// This macro returns second argument, first argument is ignored
#define OV_SCOPE_SECOND_ARG(ignored, val, ...) val

// Return macro argument value
#define OV_SCOPE_IS_ENABLED(x) OV_SCOPE_IS_ENABLED1(x)

// Generate junk macro or {0, } sequence if val is 1
#define OV_SCOPE_IS_ENABLED1(val) OV_SCOPE_IS_ENABLED2(OV_SCOPE_ARG_PLACEHOLDER_##val)

// Return second argument from possible sequences {1, 0}, {0, 1, 0}
#define OV_SCOPE_IS_ENABLED2(arg1_or_junk) OV_SCOPE_SECOND_ARG(arg1_or_junk 1, 0)

// Scope is disabled
#define OV_SCOPE_0(...)

// Scope is enabled
#define OV_SCOPE_1(...) __VA_ARGS__

#define OV_SCOPE(region, ...)           \
    OV_EXPAND(OV_CAT(OV_SCOPE_, OV_SCOPE_IS_ENABLED(OV_CAT(CC0OV_, region)))(__VA_ARGS__))

// Switch is disabled
#define OV_SWITCH_0(fn, ctx, val)

// Switch is enabled
#define OV_SWITCH_1(fn, ctx, val) internal::match<fn>(ctx, val, OV_CAT3(CC1OV_, fn, _cases));

#define OV_SWITCH(fn, ctx, val, ...)         \
    OV_EXPAND(OV_CAT(OV_SWITCH_, OV_SCOPE_IS_ENABLED(OV_CAT(CC1OV_, fn)))(fn, ctx, val))

#define OV_CASE(Case, Type) internal::make_case_wrapper<Type>(Case)

#define OV_CASE2(Case1, Case2, Type1, Type2) internal::make_case_wrapper<std::tuple<Type1, Type2>>(std::make_tuple(Case1, Case2))

#else

#define OV_SCOPE(region, ...) __VA_ARGS__

#define OV_SWITCH(fn, ctx, val, ...)    \
    internal::match<fn>(ctx, val, __VA_ARGS__);

#define OV_CASE(Case, Type) internal::make_case_wrapper<Type>(Case)

#define OV_CASE2(Case1, Case2, Type1, Type2) internal::make_case_wrapper<std::tuple<Type1, Type2>>(std::make_tuple(Case1, Case2))

#endif

}   // namespace OVConditionalCompilation
