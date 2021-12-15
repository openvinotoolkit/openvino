// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/visibility.hpp>

#ifdef OPENVINO_STATIC_LIBRARY
#    define IR_API
#    define IR_C_API
#else
#    ifdef ir_ov_frontend_EXPORTS
#        define IR_API   OPENVINO_CORE_EXPORTS
#        define IR_C_API OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS
#    else
#        define IR_API   OPENVINO_CORE_IMPORTS
#        define IR_C_API OPENVINO_EXTERN_C OPENVINO_CORE_IMPORTS
#    endif  // ir_ov_frontend_EXPORTS
#endif      // OPENVINO_STATIC_LIBRARY
