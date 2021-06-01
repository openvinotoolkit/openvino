// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/visibility.hpp"

// Increment each time when FrontEnd/InputModel/Place interface is changed
#define OV_FRONTEND_API_VERSION 1

// Defined if cmake is building the frontend_manager DLL (instead of using it)
#ifdef frontend_manager_EXPORTS
#define FRONTEND_API NGRAPH_HELPER_DLL_EXPORT
#else
#define FRONTEND_API NGRAPH_HELPER_DLL_IMPORT
#endif // frontend_manager_EXPORTS
