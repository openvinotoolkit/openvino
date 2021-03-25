// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/visibility.hpp"

// Now we use the generic helper definitions above to define NGRAPH_API
// NGRAPH_API is used for the public API symbols. It either DLL imports or DLL exports
//    (or does nothing for static build)

#ifdef _WIN32
#pragma warning(disable : 4251)
#pragma warning(disable : 4275)
#endif

#ifdef NGRAPH_STATIC_LIBRARY // defined if we are building or calling NGRAPH as a static library
#define NGRAPH_API
#else
#ifdef ngraph_EXPORTS // defined if we are building the NGRAPH DLL (instead of using it)
#define NGRAPH_API NGRAPH_HELPER_DLL_EXPORT
#else
#define NGRAPH_API NGRAPH_HELPER_DLL_IMPORT
#endif // ngraph_EXPORTS
#endif // NGRAPH_STATIC_LIBRARY

#ifndef ENABLE_UNICODE_PATH_SUPPORT
#ifdef _WIN32
#if defined __INTEL_COMPILER || defined _MSC_VER
#define ENABLE_UNICODE_PATH_SUPPORT
#endif
#elif defined(__GNUC__) && (__GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 2)) ||              \
    defined(__clang__)
#define ENABLE_UNICODE_PATH_SUPPORT
#endif
#endif
