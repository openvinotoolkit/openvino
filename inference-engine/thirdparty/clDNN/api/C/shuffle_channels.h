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

CLDNN_BEGIN_PRIMITIVE_DESC(shuffle_channels)
/// @brief The number of groups to split the channel dimension. This number must evenly divide the channel dimension size.
int32_t group;
/// @brief The index of the channel dimension (default is 1).
int32_t axis;
CLDNN_END_PRIMITIVE_DESC(shuffle_channels)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(shuffle_channels);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}

