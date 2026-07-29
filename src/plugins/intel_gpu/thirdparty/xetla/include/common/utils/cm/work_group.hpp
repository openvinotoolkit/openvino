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

/// @addtogroup xetla_util_group
/// @{

/// @brief Define a workgroup scope for a specific problem shape.
/// The size of the workgroup should <= group size.
/// @tparam size_ Is the number of subgroups within a workgroup.
template <uint32_t size_>
struct work_group_t {
private:
    uint32_t sg_id;

public:
    static constexpr uint32_t size = size_;
    __XETLA_API constexpr uint32_t get_size() { return size; }
    __XETLA_API uint32_t get_id() { return sg_id; }
    __XETLA_API void init(uint32_t id) { sg_id = id; }
    inline work_group_t(uint32_t id) : sg_id(id) {}
    inline work_group_t() = default;
    template <uint32_t scope_size>
    __XETLA_API work_group_t<scope_size> partition() {
        work_group_t<scope_size> wtile;
        wtile.init(sg_id % scope_size);
        return wtile;
    }
};
/// @} xetla_util_group
} // namespace gpu::xetla
