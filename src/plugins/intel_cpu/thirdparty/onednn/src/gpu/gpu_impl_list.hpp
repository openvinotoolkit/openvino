/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef GPU_GPU_IMPL_LIST_HPP
#define GPU_GPU_IMPL_LIST_HPP

#include "common/engine.hpp"
#include "common/impl_list_item.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

class gpu_impl_list_t {
public:
    static const impl_list_item_t *get_concat_implementation_list();
    static const impl_list_item_t *get_reorder_implementation_list(
            const memory_desc_t *src_md, const memory_desc_t *dst_md);
    static const impl_list_item_t *get_sum_implementation_list();
    static const impl_list_item_t *get_implementation_list();
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_GPU_IMPL_LIST_HPP
