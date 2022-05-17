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

#ifndef GPU_OCL_OCL_MEMORY_STORAGE_BASE_HPP
#define GPU_OCL_OCL_MEMORY_STORAGE_BASE_HPP

#include "common/memory_storage.hpp"
#include "gpu/ocl/ocl_c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

class ocl_memory_storage_base_t : public memory_storage_t {
public:
    // Explicitly define ctors due to a "circular dependencies" bug in ICC.
    ocl_memory_storage_base_t(
            engine_t *engine, const memory_storage_t *parent_storage)
        : memory_storage_t(engine, parent_storage) {}
    ocl_memory_storage_base_t(engine_t *engine)
        : ocl_memory_storage_base_t(engine, this) {}

    virtual memory_kind_t memory_kind() const = 0;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_OCL_OCL_MEMORY_STORAGE_BASE_HPP
