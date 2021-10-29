// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Defined if we are building the plugin DLL (instead of using it)
#ifdef ir_ngraph_frontend_EXPORTS
#    define IR_API OPENVINO_CORE_EXPORTS
#else
#    define IR_API OPENVINO_CORE_IMPORTS
#endif  // ir_ngraph_frontend_EXPORTS

#define IR_ASSERT(ex, msg)                 \
    {                                      \
        if (!(ex))                         \
            throw std::runtime_error(msg); \
    }

#define IR_THROW(msg) throw std::runtime_error(std::string("ERROR: ") + msg)

#define NOT_IMPLEMENTED(msg) throw std::runtime_error(std::string(msg) + " is not implemented")
