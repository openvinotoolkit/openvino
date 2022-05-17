/*******************************************************************************
* Copyright 2017-2021 Intel Corporation
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

#include "cpu/cpu_engine.hpp"

#include "common/impl_list_item.hpp"
#include "cpu/ref_concat.hpp"
#include "cpu/simple_concat.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;
#define INSTANCE_IMPL(...) \
    impl_list_item_t(impl_list_item_t::concat_type_deduction_helper_t< \
            __VA_ARGS__::pd_t>())
#define INSTANCE(...) DNNL_PRIMITIVE_IMPL(INSTANCE_IMPL, __VA_ARGS__)
// clang-format off
const impl_list_item_t cpu_concat_impl_list[] = {
        REG_CONCAT_P(INSTANCE(simple_concat_t, f32))
        REG_CONCAT_P(INSTANCE(simple_concat_t, u8))
        REG_CONCAT_P(INSTANCE(simple_concat_t, s8))
        REG_CONCAT_P(INSTANCE(simple_concat_t, s32))
        REG_CONCAT_P(INSTANCE(simple_concat_t, bf16))
        REG_CONCAT_P(INSTANCE(ref_concat_t))
        nullptr,
};
// clang-format on
#undef INSTANCE
#undef INSTANCE_IMPL
} // namespace

const impl_list_item_t *
cpu_engine_impl_list_t::get_concat_implementation_list() {
    return cpu_concat_impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
