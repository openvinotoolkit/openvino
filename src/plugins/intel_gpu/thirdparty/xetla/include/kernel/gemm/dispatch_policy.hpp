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
/// @addtogroup xetla_gemm_universal
/// @{

/// @brief Default GROUP_SWIZZLE implementation.
/// A general GROUP_SWIZZLE implementation to get an workgroup id .
/// @tparam arch_tag_ Is the HW architecture.
template <gpu_arch arch_tag_>
struct group_swizzle_default {
public:
    static constexpr gpu_arch arch_tag = arch_tag_;

    inline group_swizzle_default() = default;

    template <int idx>
    static __XETLA_API int get_tile_idx(sycl::nd_item<3> &item) {
        return item.get_group(idx);
    }
    // correct group range, nothing will be done under this swizzle policy
    static __XETLA_API void update_group_range(
            uint32_t &group_range_m, uint32_t &group_range_n) {}
};

/// @brief Default GEMM_UNIVERSAL implementation.
/// A general GEMM_UNIVERSAL implementation to provide a composition point of gemm_universal and epilogue.
/// @tparam arch_tag_ Is the HW architecture.
template <typename group_swizzle_policy_>
struct dispatch_policy_default {
    using group_swizzle_policy = group_swizzle_policy_;
    static constexpr gpu_arch arch_tag = group_swizzle_policy::arch_tag;
};

/// @brief Kslicing GEMM_UNIVERSAL implementation.
/// A special GEMM_UNIVERSAL implementation to increase the hardware occupancy by splitting the GEMM_UNIVERSAL task along k dimension.
/// It includes inter-group reduction (by using global atomic) and intra-group reduction (by using local memory for data exchange).
/// @tparam num_global_kslicing_ Is the k dim split ratio between groups.
/// @tparam num_local_kslicing_ Is the k dim split ratio within a group.
/// @tparam arch_tag_ Is the HW architecture.
template <typename group_swizzle_policy_, int global_ratio_ = 1,
        int local_ratio_ = 1>
struct dispatch_policy_kslicing {
    using group_swizzle_policy = group_swizzle_policy_;
    static constexpr int global_ratio = global_ratio_;
    static constexpr int local_ratio = local_ratio_;
    static constexpr gpu_arch arch_tag = group_swizzle_policy::arch_tag;
};
} // namespace gpu::xetla::kernel
