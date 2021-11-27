// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/frontend_exceptions.hpp"

#ifdef OPENVINO_STATIC_LIBRARY
#    define TF_API
#    define TF_C_API
#else
#    ifdef tensorflow_ov_frontend_EXPORTS
#        define TF_API   OPENVINO_CORE_EXPORTS
#        define TF_C_API OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS
#    else
#        define TF_API   OPENVINO_CORE_IMPORTS
#        define TF_C_API OPENVINO_EXTERN_C OPENVINO_CORE_IMPORTS
#    endif  // tensorflow_ov_frontend_EXPORTS
#endif      // OPENVINO_STATIC_LIBRARY
