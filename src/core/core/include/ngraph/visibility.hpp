// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// https://gcc.gnu.org/wiki/Visibility
// Generic helper definitions for shared library support
#if defined _WIN32 || defined __CYGWIN__
#define NGRAPH_HELPER_DLL_IMPORT __declspec(dllimport)
#define NGRAPH_HELPER_DLL_EXPORT __declspec(dllexport)
#define NGRAPH_HELPER_DLL_LOCAL
#elif defined(__GNUC__) && __GNUC__ >= 4
#define NGRAPH_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#define NGRAPH_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#define NGRAPH_HELPER_DLL_LOCAL __attribute__((visibility("hidden")))
#else
#define NGRAPH_HELPER_DLL_IMPORT
#define NGRAPH_HELPER_DLL_EXPORT
#define NGRAPH_HELPER_DLL_LOCAL
#endif
