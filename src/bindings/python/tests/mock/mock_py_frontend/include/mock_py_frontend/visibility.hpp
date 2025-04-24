// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"

#ifdef openvino_mock_py_frontend_EXPORTS
#    define MOCK_API OPENVINO_CORE_EXPORTS
#else
#    define MOCK_API OPENVINO_CORE_IMPORTS
#endif  // openvino_mock_py_frontend_EXPORTS

#define MOCK_C_API OPENVINO_EXTERN_C MOCK_API
