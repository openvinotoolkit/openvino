// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/visibility.hpp"

#define OV_FRONTEND_API_VERSION 1 // Increment each time when FrontEnd/InputModel/Place interface is changed

#ifdef frontend_manager_EXPORTS // defined if cmake is building the frontend_manager DLL (instead of using it)
#define FRONTEND_API NGRAPH_HELPER_DLL_EXPORT
#else
#define FRONTEND_API NGRAPH_HELPER_DLL_IMPORT
#endif // frontend_manager_EXPORTS
