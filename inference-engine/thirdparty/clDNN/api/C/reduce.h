/*
// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "cldnn.h"

/// @addtogroup c_api C API
/// @{
/// @addtogroup c_topology Network Topology
/// @{
/// @addtogroup c_primitives Primitives
/// @{

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    cldnn_reduce_along_b = 0,
    cldnn_reduce_along_f = CLDNN_TENSOR_BATCH_DIM_MAX,
    cldnn_reduce_along_x = CLDNN_TENSOR_BATCH_DIM_MAX + CLDNN_TENSOR_FEATURE_DIM_MAX,
    cldnn_reduce_along_y = cldnn_reduce_along_x + 1,
    cldnn_reduce_along_z = cldnn_reduce_along_y + 1,
    cldnn_reduce_along_w = cldnn_reduce_along_z + 1
} cldnn_reduce_axis;

// @brief Select mode for reduce layer ( @CLDNN_PRIMITIVE_DESC{reduce} ​).
typedef enum {
    /// @brief Reduce max
    cldnn_reduce_max,
    /// @brief Reduce min
    cldnn_reduce_min,
    /// @brief Reduce mean
    cldnn_reduce_mean,
    /// @brief Reduce prod
    cldnn_reduce_prod,
    /// @brief Reduce sum
    cldnn_reduce_sum,
    /// @brief Reduce and
    cldnn_reduce_and,
    /// @brief Reduce or
    cldnn_reduce_or,
    /// @brief Reduce sum square
    cldnn_reduce_sum_square,
    /// @brief Reduce l1
    cldnn_reduce_l1,
    /// @brief Reduce l2
    cldnn_reduce_l2,
    /// @brief Reduce log sum
    cldnn_reduce_log_sum,
    /// @brief Reduce log sum exp
    cldnn_reduce_log_sum_exp
} cldnn_reduce_mode;

CLDNN_BEGIN_PRIMITIVE_DESC(reduce)
/// @brief Keep the reduced dimension or not, 1 mean keep reduced dimension
int32_t keep_dims;
/// @brief Reduce operation type
int32_t mode;
/// @brief List of axes to reduce
cldnn_uint16_t_arr axes;
CLDNN_END_PRIMITIVE_DESC(reduce)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(reduce);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}

