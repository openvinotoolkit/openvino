// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/visibility.hpp"

#ifdef TRANSFORMATIONS_DLL_EXPORTS // defined if we are building the transformations library
#define TRANSFORMATIONS_API NGRAPH_HELPER_DLL_EXPORT
#else
#define TRANSFORMATIONS_API NGRAPH_HELPER_DLL_IMPORT
#endif
