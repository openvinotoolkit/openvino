// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined (HAVE_SSE) || defined (HAVE_AVX2)
#if defined (_WIN32)
#include <emmintrin.h>
#else
#include <x86intrin.h>
#endif
#endif

#if defined (WIN32) || defined (_WIN32)
#if defined (__INTEL_COMPILER)
#define DLSDK_EXT_IVDEP() __pragma(ivdep)
#elif defined(_MSC_VER)
#define DLSDK_EXT_IVDEP() __pragma(loop(ivdep))
#else
#define DLSDK_EXT_IVDEP()
#endif
#elif defined(__linux__)
#if defined(__INTEL_COMPILER)
#define DLSDK_EXT_IVDEP() _Pragma("ivdep")
#elif defined(__GNUC__)
#define DLSDK_EXT_IVDEP() _Pragma("GCC ivdep")
#else
#define DLSDK_EXT_IVDEP()
#endif
#else
#define DLSDK_EXT_IVDEP()
#endif