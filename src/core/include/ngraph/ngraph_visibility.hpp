// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/visibility.hpp"
#include "openvino/core/core_visibility.hpp"

#define NGRAPH_API      OPENVINO_API
#define NGRAPH_API_C    OPENVINO_API_C
#define NGRAPH_EXTERN_C OPENVINO_EXTERN_C

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    define ENABLE_UNICODE_PATH_SUPPORT
#endif
