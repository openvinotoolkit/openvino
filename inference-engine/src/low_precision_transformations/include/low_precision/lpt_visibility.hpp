// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/visibility.hpp"

/**
 * @file lpt_visibility.hpp
 * @brief Defines visibility settings for Inference Engine LP Transformations library
 */

#ifdef inference_engine_lp_transformations_EXPORTS
#define LP_TRANSFORMATIONS_API NGRAPH_HELPER_DLL_EXPORT
#else
#define LP_TRANSFORMATIONS_API NGRAPH_HELPER_DLL_IMPORT
#endif
