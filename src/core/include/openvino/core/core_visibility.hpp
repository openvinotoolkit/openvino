// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"

#define OV_NEW_API 1
// Now we use the generic helper definitions above to define OPENVINO_API
// OPENVINO_API is used for the public API symbols. It either DLL imports or DLL exports
//    (or does nothing for static build)

#ifdef _WIN32
#    pragma warning(disable : 4251)
#    pragma warning(disable : 4275)
#endif

#ifdef OPENVINO_STATIC_LIBRARY  // defined if we are building or calling NGRAPH as a static library
#    define OPENVINO_API
#    define OPENVINO_API_C(...) __VA_ARGS__
#else
#    ifdef IMPLEMENT_OPENVINO_API  // defined if we are building the NGRAPH DLL (instead of using it)
#        define OPENVINO_API        OPENVINO_CORE_EXPORTS
#        define OPENVINO_API_C(...) OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS __VA_ARGS__ OPENVINO_CDECL
#    else
#        define OPENVINO_API        OPENVINO_CORE_IMPORTS
#        define OPENVINO_API_C(...) OPENVINO_EXTERN_C OPENVINO_CORE_IMPORTS __VA_ARGS__ OPENVINO_CDECL
#    endif  // IMPLEMENT_OPENVINO_API
#endif      // OPENVINO_STATIC_LIBRARY
