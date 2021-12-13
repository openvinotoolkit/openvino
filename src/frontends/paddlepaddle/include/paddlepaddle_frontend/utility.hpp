// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/frontend/common/frontend_manager.hpp>

#ifdef OPENVINO_STATIC_LIBRARY
#    define PDPD_API
#    define PDPD_C_API
#else
#    ifdef paddlepaddle_ov_frontend_EXPORTS
#        define PDPD_API   OPENVINO_CORE_EXPORTS
#        define PDPD_C_API OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS
#    else
#        define PDPD_API   OPENVINO_CORE_IMPORTS
#        define PDPD_C_API OPENVINO_EXTERN_C OPENVINO_CORE_IMPORTS
#    endif  // paddlepaddle_ov_frontend_EXPORTS
#endif      // OPENVINO_STATIC_LIBRARY

#define PDPD_ASSERT(ex, msg)               \
    {                                      \
        if (!(ex))                         \
            throw std::runtime_error(msg); \
    }

#define PDPD_THROW(msg) throw std::runtime_error(std::string("ERROR: ") + msg)

#define NOT_IMPLEMENTED(msg) throw std::runtime_error(std::string(msg) + " is not implemented")
