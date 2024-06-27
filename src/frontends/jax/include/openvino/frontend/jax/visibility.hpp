// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/visibility.hpp"

#ifdef OPENVINO_STATIC_LIBRARY
#    define JAX_API
#    define JAX_C_API
#else
#    ifdef openvino_jax_frontend_EXPORTS
#        define JAX_API   OPENVINO_CORE_EXPORTS
#        define JAX_C_API OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS
#    else
#        define JAX_API   OPENVINO_CORE_IMPORTS
#        define JAX_C_API OPENVINO_EXTERN_C OPENVINO_CORE_IMPORTS
#    endif  // openvino_jax_frontend_EXPORTS
#endif      // OPENVINO_STATIC_LIBRARY
