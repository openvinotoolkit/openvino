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

#ifdef _WIN32
#include "../../../common/utils/cm/common.hpp"
#else
#include "common/utils/cm/common.hpp"
#endif

namespace gpu::xetla {

/// @addtogroup xetla_util_named_barrier
/// @{

enum class nbarrier_role : uint8_t {
    producer_consumer = 0,
    producer = 1,
    consumer = 2
};

///
/// @brief xetla nbarrier definition API.
///  This is the API to define a named barrier within subgroup.
/// @tparam num_producers is the number of subgroups participating the barrier as producer.
/// @tparam num_consumers is the number of subgroups participating the barrier as consumer.
///
template <uint8_t num_producers = 1, uint8_t num_consumers = 1,
        gpu_arch arch_tag = gpu_arch::Xe>
struct xetla_nbarrier_t {
    ///
    /// @brief Description of named barrier objection.
    ///
    xetla_vector<uint32_t, 16> nbar;
    uint32_t barrier_id;

    /// @param role is the role of subgroup when participating the barrier.
    /// @param nbarrier_id [in] is the id of the barrier.
    /// note:  all subgroups participating the barrier should have the same barrier_id.
    __XETLA_API void init_nbarrier(uint8_t nbarrier_id,
            nbarrier_role role = nbarrier_role::producer_consumer) {
        nbar[2] = (uint32_t)nbarrier_id | uint32_t((uint8_t)role << 14)
                | uint32_t(num_producers << 16) | uint32_t(num_consumers << 24);
        barrier_id = nbarrier_id;
    }

    /// @brief named barrier signal from subgroup.
    /// @param bar is the named barrier object.
    ///
    __XETLA_API void arrive() {
        constexpr uint32_t sfid = 0x3;
        constexpr uint32_t exDesc = sfid;
        constexpr uint32_t msg_desc = 0x2000004;
        constexpr uint32_t execSize = 0;

        xetla_raw_send<uint32_t, 16, execSize, sfid, 1>(nbar, exDesc, msg_desc);
    }

    /// @brief named barrier wait within subgroup.
    ///
    __XETLA_API void wait() { named_barrier_wait(barrier_id); }

    /// @brief named barrier signal from subgroup.
    ///
    __XETLA_API void arrive_wait() {
        arrive();
        wait();
    }
};

/// @} xetla_util_named_barrier

} // namespace gpu::xetla
