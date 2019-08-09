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

/// @brief Performs deformable convolution on a preprocessed data. Should be created after deformable_interp primitive.
CLDNN_BEGIN_PRIMITIVE_DESC(deformable_conv)
/// @brief On how many cards split the computation to.
uint32_t split;
/// @brief User-defined output data size of the primitive (w/o padding).
cldnn_tensor output_size;
/// @brief Array of primitive ids containing weights data. Size of array should be equivalent to @p split.
cldnn_primitive_id_arr weights;
/// @brief Array of primitive ids containing bias data. Size of array should be equivalent to @p split.
cldnn_primitive_id_arr bias;
/// @brief Number of feature groups (grouped convolution). If more than 1 then weights/bias count needs to be 1.
uint32_t groups;

CLDNN_END_PRIMITIVE_DESC(deformable_conv)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(deformable_conv);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}

