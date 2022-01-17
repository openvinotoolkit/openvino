// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"

#ifdef OPENVINO_STATIC_LIBRARY
#    define MOCK_API
#else
#    ifdef IMPLEMENT_OPENVINO_API
#        define MOCK_API OPENVINO_CORE_EXPORTS
#    else
#        define MOCK_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#endif      // OPENVINO_STATIC_LIBRARY
