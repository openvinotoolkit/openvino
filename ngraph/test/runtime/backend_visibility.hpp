// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/visibility.hpp"

#ifdef ngraph_backend_EXPORTS // defined if we are building the ngraph_backend as shared library
#define BACKEND_API NGRAPH_HELPER_DLL_EXPORT
#else
#define BACKEND_API NGRAPH_HELPER_DLL_IMPORT
#endif
