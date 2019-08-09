/*
// Copyright (c) 2016 Intel Corporation
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

#include "api/C/cldnn.h"

namespace cldnn {
/// @brief Changes how data is ordered in memory. Value type is not changed & all information is preserved.
/// @details Corresponding values are bitwise equal before/after reorder.
/// Also merged with subtraction layer, which can subtract values while doing reordering.
CLDNN_BEGIN_PRIMITIVE_DESC(generic_layer)
/// @brief Requested memory layout.
cldnn_layout output_layout;
const void* generic_params;

CLDNN_END_PRIMITIVE_DESC(generic_layer)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(generic_layer);

}  // namespace cldnn