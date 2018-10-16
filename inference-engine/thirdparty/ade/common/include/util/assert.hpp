// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef UTIL_ASSERT_HPP
#define UTIL_ASSERT_HPP

#include <cassert>
#include <utility>

#if !defined(__EXCEPTIONS)
#include <stdlib.h>
#include <stdio.h>
#endif

#if defined(_MSC_VER)
#define unreachable() __assume(0)
#elif defined(__GNUC__)
#define unreachable() __builtin_unreachable()
#else
#define unreachable() do{}while(false)
#endif

#ifdef DEVELOPMENT_MODE
#include <cstdio>
#include <cstdlib> // abort()
inline void dev_assert(bool val, const char* str, int line, const char* file)
{
    if (!val)
    {
        fprintf(stderr, "%s:%d: Assertion \"%s\" failed\n", file, line, str);
        fflush(stderr);
        abort();
    }
}
#define ASSERT(arg) dev_assert(static_cast<bool>(arg), #arg, __LINE__, __FILE__)
#else
#define ASSERT(arg) do {                         \
    constexpr bool _assert_tmp = false && (arg); \
    (void) _assert_tmp;                          \
    assert(arg);                                 \
} while(false)
#endif

/// Stronger version of assert which translates to compiler hint in optimized
/// builds. Do not use it if you have subsequent error recovery code because
/// this code can be optimized away.
/// Expression is always evaluated, avoid functions calls in it.
/// Static analyzers friendly and can silence "possible null pointer
/// dereference" warnings.
#if defined(DEVELOPMENT_MODE) || !defined(NDEBUG)
#define ASSERT_STRONG(expr) ASSERT(expr)
#else
#define ASSERT_STRONG(expr) do{ if(!(expr)) { unreachable(); } }while(false)
#endif

#define UNREACHABLE(str) ASSERT_STRONG(!str)

template <class ExceptionType>
[[noreturn]] void throw_error(ExceptionType &&e)
{
#if defined(__EXCEPTIONS) || defined(_CPPUNWIND)
    throw std::forward<ExceptionType>(e);
#else
    fprintf(stderr, "An exception thrown! %s\n" , e.what());
    fflush(stderr);
    abort();
#endif
}

#if (defined NDEBUG) && !(defined DEVELOPMENT_MODE)
#define ASSERT_OR_THROW_ON_FAIL(expr) if (!expr) throw_error(std::runtime_error("Assertion failed!"))
#else
#define ASSERT_OR_THROW_ON_FAIL(expr) ASSERT(expr);
#endif

#endif // UTIL_ASSERT_HPP
