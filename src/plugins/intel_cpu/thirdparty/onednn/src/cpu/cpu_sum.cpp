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

#include "common/impl_list_item.hpp"

#include "cpu/cpu_engine.hpp"

#include "cpu/ref_sum.hpp"
#include "cpu/simple_sum.hpp"

#if DNNL_X64
#include "cpu/x64/jit_avx512_core_bf16_sum.hpp"
using namespace dnnl::impl::cpu::x64;
#endif

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;

#define INSTANCE_IMPL(...) \
    impl_list_item_t(impl_list_item_t::sum_type_deduction_helper_t< \
            __VA_ARGS__::pd_t>())
#define INSTANCE(...) DNNL_PRIMITIVE_IMPL(INSTANCE_IMPL, __VA_ARGS__)
#define INSTANCE_X64(...) DNNL_X64_ONLY(INSTANCE(__VA_ARGS__))
// clang-format off
const impl_list_item_t cpu_sum_impl_list[] = {
        REG_SUM_P(INSTANCE_X64(jit_bf16_sum_t, bf16, bf16))
        REG_SUM_P(INSTANCE_X64(jit_bf16_sum_t, bf16, f32))
        REG_SUM_P(INSTANCE(simple_sum_t, bf16))
        REG_SUM_P(INSTANCE(simple_sum_t, bf16, f32))
        REG_SUM_P(INSTANCE(simple_sum_t, f32))
        REG_SUM_P(INSTANCE(ref_sum_t))
        nullptr,
};
// clang-format on
#undef INSTANCE_X64
#undef INSTANCE
#undef INSTANCE_IMPL
} // namespace

const impl_list_item_t *cpu_engine_impl_list_t::get_sum_implementation_list() {
    return cpu_sum_impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
