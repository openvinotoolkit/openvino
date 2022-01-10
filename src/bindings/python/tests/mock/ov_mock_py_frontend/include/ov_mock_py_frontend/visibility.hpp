// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/visibility.hpp"
#include "openvino/frontend/visibility.hpp"

// Defined if we are building the plugin DLL (instead of using it)
#ifdef ov_mock_py_frontend_EXPORTS
#    define MOCK_API OPENVINO_CORE_EXPORTS
#else
#    define MOCK_API OPENVINO_CORE_IMPORTS
#endif  // ov_mock_py_frontend_EXPORTS
