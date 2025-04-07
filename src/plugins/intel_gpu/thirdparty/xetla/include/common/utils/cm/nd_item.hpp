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

namespace sycl {

/// @brief  The item struct to explict identify the group / local id information
///
/// @tparam dims The dimension for the execution item
///
template <int dims = 1>
class nd_item {
public:
    inline nd_item() = default;
    /// @brief Returns the group id for the requested dimensions.
    ///
    /// @param dimension The requested dimension
    /// @return uint32_t The group id
    ///
    inline uint32_t get_group(int dimension) const {
        return cm_group_id(dims - 1 - dimension);
    }

    /// @brief Returns local id in the work group for the requested dimensions
    ///
    /// @param dimension The requested dimension
    /// @return uint32_t The local id
    ///
    inline uint32_t get_local_id(int dimension) const {
        return cm_local_id(dims - 1 - dimension);
    }

    /// @brief Returns local size in the work group for the requested dimensions
    ///
    /// @param dimension The requested dimension
    /// @return uint32_t The local size
    ///
    inline uint32_t get_local_range(int dimension) const {
        return cm_local_size(dims - 1 - dimension);
    }

    /// @brief Returns local linear id in the work group
    /// @note The right-most term in the object’s range varies fastest in the linearization.
    /// @return uint32_t The local linear id
    ///
    inline uint32_t get_local_linear_id() const {
        if constexpr (dims == 1) { return get_local_id(0); }
        if constexpr (dims == 2) {
            return get_local_id(1) + get_local_id(0) * get_local_range(1);
        }
        if constexpr (dims == 3) {
            return get_local_id(2) + get_local_id(1) * get_local_range(2)
                    + get_local_id(0) * get_local_range(1) * get_local_range(2);
        }
    }

    /// @brief Returns global linear id
    ///
    /// @return uint32_t The global linear id
    ///
    inline uint32_t get_global_linear_id() const {
        return cm_linear_global_id();
    }

    inline uint32_t get_group_range(int dimension) const {
        return cm_group_count(dims - 1 - dimension);
    }

    inline uint32_t get_group_linear_id() const { return cm_linear_group_id(); }
};

} // namespace sycl
