// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/visibility.hpp"

#ifdef inference_engine_transformations_EXPORTS
#define TRANSFORMATIONS_API NGRAPH_HELPER_DLL_EXPORT
#else
#define TRANSFORMATIONS_API NGRAPH_HELPER_DLL_IMPORT
#endif
