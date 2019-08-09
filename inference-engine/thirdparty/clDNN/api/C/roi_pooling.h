/*
// Copyright (c) 2017 Intel Corporation
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
#include <stdbool.h>
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

CLDNN_BEGIN_PRIMITIVE_DESC(roi_pooling)
/// @brief Pooling method. See #cldnn_pooling_mode.
int32_t mode;
/// @brief True, if pooling is position sensitive (PSROIPoolng).
bool position_sensitive;
/// @brief Output width.
int pooled_width;
/// @brief Output height.
int pooled_height;
/// @brief Count of sub bins in x spatial dimension.
int spatial_bins_x;
/// @brief Count of sub bins in y spatial dimension.
int spatial_bins_y;
/// @brief Output features count (applied for position sensitive case only).
int output_dim;
/// @brief Transformation parameter.
float trans_std;
/// @brief False, if pooling is deformable (DeformablePSROIPoolng).
bool no_trans;
/// @brief Ratio of the coordinates used in RoIs to the width (and height) of the input data.
float spatial_scale;
/// @brief Size of pooled part.
int part_size;
/// @brief Size of pooled group.
int group_size;
CLDNN_END_PRIMITIVE_DESC(roi_pooling)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(roi_pooling);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}

