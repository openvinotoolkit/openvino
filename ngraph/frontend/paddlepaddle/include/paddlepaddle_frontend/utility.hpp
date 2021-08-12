// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_manager.hpp>

// Defined if we are building the plugin DLL (instead of using it)
#ifdef paddlepaddle_ngraph_frontend_EXPORTS
#define PDPD_API NGRAPH_HELPER_DLL_EXPORT
#else
#define PDPD_API NGRAPH_HELPER_DLL_IMPORT
#endif // paddlepaddle_ngraph_frontend_EXPORTS

#define PDPD_ASSERT(ex, msg)                                                                       \
    {                                                                                              \
        if (!(ex))                                                                                 \
            throw std::runtime_error(msg);                                                         \
    }

#define PDPD_THROW(msg) throw std::runtime_error(std::string("ERROR: ") + msg)

#define NOT_IMPLEMENTED(msg) throw std::runtime_error(std::string(msg) + " is not implemented")
