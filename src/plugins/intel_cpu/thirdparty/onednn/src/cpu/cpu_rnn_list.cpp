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

#include "cpu/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;

// clang-format off
const impl_list_item_t impl_list[] = {
        REG_RNN_P_FWD(CPU_INSTANCE(ref_rnn_fwd_f32_t))
        REG_RNN_P_FWD(CPU_INSTANCE(ref_rnn_fwd_bf16_t))
#ifdef ENABLE_UNUSED_PRIM
        REG_RNN_P_FWD(CPU_INSTANCE(ref_rnn_fwd_s8s8_t))
        REG_RNN_P_FWD(CPU_INSTANCE(ref_rnn_fwd_u8s8_t))
        REG_RNN_P_BWD(CPU_INSTANCE(ref_rnn_bwd_f32_t))
        REG_RNN_P_BWD(CPU_INSTANCE(ref_rnn_bwd_bf16_t))
#endif
        /* eol */
        nullptr,
};
// clang-format on
} // namespace

const impl_list_item_t *get_rnn_impl_list(const rnn_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
