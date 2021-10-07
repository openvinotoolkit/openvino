// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_exceptions.hpp>

#ifdef tensorflow_ngraph_frontend_EXPORTS
#    define TF_API NGRAPH_HELPER_DLL_EXPORT
#else
#    define TF_API NGRAPH_HELPER_DLL_IMPORT
#endif  // tensorflow_ngraph_frontend_EXPORTS

#define TF_ASSERT(ex, msg)                 \
    {                                      \
        if (!(ex))                         \
            throw std::runtime_error(msg); \
    }

#define TF_THROW(msg) throw std::runtime_error(std::string("ERROR: ") + msg)

#define NOT_IMPLEMENTED(msg) throw std::runtime_error(std::string(msg) + " is not implemented")

#define NGRAPH_VLOG(I) std::ostringstream()
