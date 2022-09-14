// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"

#ifdef IMPLEMENT_OPENVINO_API
#    define MOCK_API OPENVINO_CORE_EXPORTS
#else
#    define MOCK_API OPENVINO_CORE_IMPORTS
#endif  // IMPLEMENT_OPENVINO_API

#if defined _WIN32
#    define MOCK_C_API OPENVINO_EXTERN_C
#else
#    define MOCK_C_API OPENVINO_EXTERN_C MOCK_API
#endif
