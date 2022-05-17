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

#include "gpu/gpu_impl_list.hpp"

#include "common/impl_list_item.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_sum_pd.hpp"
#include "gpu/jit/gen9_simple_sum.hpp"
#include "gpu/ocl/gen9_sum.hpp"
#include "gpu/ocl/ref_sum.hpp"
#include "gpu/ocl/simple_sum.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

namespace {
// TODO: Re-enable nGEN-based implementation after architecture
// dispatching is implemented.
// INSTANCE(jit::gen9_simple_sum_t),
#define INSTANCE(...) \
    impl_list_item_t(impl_list_item_t::sum_type_deduction_helper_t< \
            __VA_ARGS__::pd_t>())
const impl_list_item_t sum_impl_list[] = {
        INSTANCE(ocl::gen9_sum_t),
        INSTANCE(ocl::simple_sum_t<data_type::f32>),
        INSTANCE(ocl::ref_sum_t),
        nullptr,
};
#undef INSTANCE
} // namespace

const impl_list_item_t *gpu_impl_list_t::get_sum_implementation_list() {
    return sum_impl_list;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
