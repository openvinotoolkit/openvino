// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef GMOCK_INCLUDE_GMOCK_INTERNAL_GMOCK_PORT_H_
#define GMOCK_INCLUDE_GMOCK_INTERNAL_GMOCK_PORT_H_

#include <assert.h>
#include <stdlib.h>
#include <iostream>

// Most of the utilities needed for porting Google Mock are also
// required for Google Test and are defined in gtest-port.h.
//
// Note to maintainers: to reduce code duplication, prefer adding
// portability utilities to Google Test's gtest-port.h instead of
// here, as Google Mock depends on Google Test.  Only add a utility
// here if it's truly specific to Google Mock.
#include "gtest/internal/gtest-linked_ptr.h"
#include "gtest/internal/gtest-port.h"
#include "gmock/internal/custom/gmock-port.h"

// To avoid conditional compilation everywhere, we make it
// gmock-port.h's responsibility to #include the header implementing
// tr1/tuple.  gmock-port.h does this via gtest-port.h, which is
// guaranteed to pull in the tuple header.

// For MS Visual C++, check the compiler version. At least VS 2003 is
// required to compile Google Mock.
#if defined(_MSC_VER) && _MSC_VER < 1310
# error "At least Visual C++ 2003 (7.1) is required to compile Google Mock."
#endif

// Macro for referencing flags.  This is public as we want the user to
// use this syntax to reference Google Mock flags.
#define GMOCK_FLAG(name) FLAGS_gmock_##name

#if !defined(GMOCK_DECLARE_bool_)

// Macros for declaring flags.
#define GMOCK_DECLARE_bool_(name) extern GTEST_API_ bool GMOCK_FLAG(name)
#define GMOCK_DECLARE_int32_(name) \
    extern GTEST_API_ ::testing::internal::Int32 GMOCK_FLAG(name)
#define GMOCK_DECLARE_string_(name) \
    extern GTEST_API_ ::std::string GMOCK_FLAG(name)

// Macros for defining flags.
#define GMOCK_DEFINE_bool_(name, default_val, doc) \
    GTEST_API_ bool GMOCK_FLAG(name) = (default_val)
#define GMOCK_DEFINE_int32_(name, default_val, doc) \
    GTEST_API_ ::testing::internal::Int32 GMOCK_FLAG(name) = (default_val)
#define GMOCK_DEFINE_string_(name, default_val, doc) \
    GTEST_API_ ::std::string GMOCK_FLAG(name) = (default_val)

#endif  // !defined(GMOCK_DECLARE_bool_)

#endif  // GMOCK_INCLUDE_GMOCK_INTERNAL_GMOCK_PORT_H_
