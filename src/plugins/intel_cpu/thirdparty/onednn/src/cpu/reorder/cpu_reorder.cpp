/*******************************************************************************
* Copyright 2017-2021 Intel Corporation
* Copyright 2020 FUJITSU LIMITED
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

/* regular reorders */
const std::map<reorder_impl_key_t, const void *> regular_impl_list_map {
        {{f32, bf16, 0}, &regular_f32_bf16_impl_list_map},
        {{f32, f16, 0}, &regular_f32_f16_impl_list_map},
        {{f32, f32, 0}, &regular_f32_f32_impl_list_map},
        {{f32, s32, 0}, &regular_f32_s32_impl_list_map},
        {{f32, s8, 0}, &regular_f32_s8_impl_list_map},
        {{f32, u8, 0}, &regular_f32_u8_impl_list_map},
        {{f32, bin, 0}, &regular_f32_bin_impl_list_map},
        {{bf16, data_type::undef, 0}, &regular_bf16_impl_list_map},
        {{f16, data_type::undef, 0}, &regular_f16_impl_list_map},
        {{s32, data_type::undef, 0}, &regular_s32_impl_list_map},
        {{s8, data_type::undef, 0}, &regular_s8_impl_list_map},
        {{u8, data_type::undef, 0}, &regular_u8_impl_list_map},
        {{bin, data_type::undef, 0}, &regular_bin_impl_list_map},
};

/* conv reorders w/ compensation */
const std::map<reorder_impl_key_t, const void *> comp_s8s8_impl_list_map {
        {{f32, s8, 0}, &comp_f32_s8_impl_list_map},
        {{bf16, s8, 0}, &comp_bf16_s8_impl_list_map},
        {{s8, s8, 0}, &comp_s8_s8_impl_list_map},
};

const impl_list_item_t *cpu_engine_impl_list_t::get_reorder_implementation_list(
        const memory_desc_t *src_md, const memory_desc_t *dst_md) {
    reorder_impl_key_t dt_pair {src_md->data_type, dst_md->data_type, 0};
    const bool do_comp_s8s8 = dst_md->extra.flags
            & (memory_extra_flags::compensation_conv_s8s8
                    | memory_extra_flags::compensation_conv_asymmetric_src);
    const auto &map = do_comp_s8s8 ? comp_s8s8_impl_list_map : regular_impl_list_map;

    static const impl_list_item_t empty_list[] = {nullptr};

    auto iter = map.find(dt_pair);
    if (iter == map.end()) {
        dt_pair.dst_dt = data_type::undef;
        iter = map.find(dt_pair);
        if (iter == map.end()) return empty_list;
    }

    const impl_list_map_t *p_impl_list = (const impl_list_map_t *)iter->second;

    reorder_impl_key_t key {dt_pair.src_dt, dt_pair.dst_dt, src_md->ndims};

    {
        const auto it = p_impl_list->find(key);
        if (it != p_impl_list->cend()) return it->second.data();
    }

    {
        key.ndims = 0;
        const auto it = p_impl_list->find(key);
        if (it != p_impl_list->cend()) return it->second.data();
    }

    return empty_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
