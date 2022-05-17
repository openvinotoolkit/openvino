/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "cpu/reorder/cpu_reorder.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// clang-format off

const impl_list_map_t regular_bin_impl_list_map {
    // bin ->
    {{bin, data_type::undef, 4}, {
        REG_REORDER_P(REG_SR_DIRECT_COPY(bin, bin))

        REG_REORDER_P(REG_SR(bin, any, bin, OIhw8o32i, fmt_order_keep))

        REG_REORDER_P(REG_SR(bin, any, bin, OIhw16o32i, fmt_order_keep))

        REG_REORDER_P(REG_SR_BIDIR(u8, any, u8, nChw8c))

        nullptr,
    }},
};

// clang-format on

} // namespace cpu
} // namespace impl
} // namespace dnnl
