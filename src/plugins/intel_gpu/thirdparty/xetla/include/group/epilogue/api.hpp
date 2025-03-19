/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @file
/// C++ API

#pragma once

#include "group/epilogue/common.hpp"

namespace gpu::xetla::group {

/// @addtogroup xetla_epilogue
/// @{

/// @brief Is the epilogue functor.
/// @tparam epilogue_policy Is the epilogue policy.
/// @tparam tile_shape Is the workgroup-level tile shape.
/// @tparam mem_desc_c Is the memory descriptor of matC.
template <typename epilogue_policy, typename tile_shape, typename mem_desc_c,
        class enable = void>
class epilogue_t {};

/// @} xetla_epilogue

} // namespace gpu::xetla::group
