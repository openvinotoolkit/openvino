/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include "common/bfloat16.hpp"
#include "cpu/ref_shuffle.hpp"

#if DNNL_X64
#include "cpu/x64/shuffle/jit_uni_shuffle.hpp"
using namespace dnnl::impl::cpu::x64;
#endif

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;

// clang-format off
const impl_list_item_t impl_list[] = {
#ifdef ENABLE_UNUSED_PRIM
        REG_SHUFFLE_P(CPU_INSTANCE_X64(jit_uni_shuffle_t, avx512_common))
        REG_SHUFFLE_P(CPU_INSTANCE_X64(jit_uni_shuffle_t, avx))
        REG_SHUFFLE_P(CPU_INSTANCE_X64(jit_uni_shuffle_t, sse41))
        REG_SHUFFLE_P(CPU_INSTANCE(ref_shuffle_t))
#endif
        /* eol */
        nullptr,
};
// clang-format on
} // namespace

const impl_list_item_t *get_shuffle_impl_list(const shuffle_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
