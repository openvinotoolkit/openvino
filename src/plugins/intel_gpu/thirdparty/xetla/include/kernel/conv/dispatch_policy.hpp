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

#include "kernel/gemm/common.hpp"

namespace gpu::xetla::kernel {
/// @addtogroup xetla_conv
/// @{

/// @brief cslicing conv implementation.
/// A special conv implementation to increase the hardware occupancy by splitting the conv task along input channels.
/// @tparam local_ratio_ Is the c dim split ratio within a group.
/// @tparam local_slicing_mem_space Is the type of memory used for local slicing. mem_space::local is for SLM,
/// mem_space::global for scratchpad memory.
/// @tparam arch_tag_ Is the HW architecture.
template <typename group_swizzle_policy_, int global_ratio_ = 1,
        int local_ratio_ = 1,
        mem_space local_slicing_mem_space_ = mem_space::local>
struct dispatch_policy_slicing {
    using group_swizzle_policy = group_swizzle_policy_;
    static constexpr int local_ratio = local_ratio_;
    static constexpr mem_space local_slicing_mem_space
            = local_slicing_mem_space_;
    static constexpr int global_ratio = global_ratio_;
    static constexpr gpu_arch arch_tag = group_swizzle_policy::arch_tag;
};

/// @} xetla_conv

} // namespace gpu::xetla::kernel
