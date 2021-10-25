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

#define NGRAPH_VLOG(I) std::ostringstream()
