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

#include <CL/cl.h>

#include "oneapi/dnnl/dnnl_ocl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/memory.hpp"
#include "common/utils.hpp"

#include "gpu/ocl/ocl_c_types_map.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::gpu::ocl;

status_t dnnl_ocl_interop_memory_create(memory_t **memory,
        const memory_desc_t *md, engine_t *engine, memory_kind_t memory_kind,
        void *handle) {

    bool ok = !utils::any_null(memory, md, engine)
            && engine->runtime_kind() == runtime_kind::ocl;
    if (!ok) return status::invalid_arguments;

    const auto mdw = memory_desc_wrapper(md);
    if (mdw.format_any() || mdw.has_runtime_dims_or_strides())
        return status::invalid_arguments;

    size_t size = mdw.size();
    unsigned flags = (handle == DNNL_MEMORY_ALLOCATE)
            ? memory_flags_t::alloc
            : memory_flags_t::use_runtime_ptr;
    void *handle_ptr = (handle == DNNL_MEMORY_ALLOCATE) ? nullptr : handle;

    bool is_usm = memory_kind == memory_kind::usm;

    std::unique_ptr<memory_storage_t> mem_storage;
    if (is_usm)
        mem_storage.reset(new ocl_usm_memory_storage_t(engine));
    else
        mem_storage.reset(new ocl_buffer_memory_storage_t(engine));
    if (!mem_storage) return status::out_of_memory;

    CHECK(mem_storage->init(flags, size, handle_ptr));

    return safe_ptr_assign(
            *memory, new memory_t(engine, md, std::move(mem_storage)));
}

status_t dnnl_ocl_interop_memory_get_mem_object(
        const memory_t *memory, cl_mem *mem_object) {
    if (utils::any_null(mem_object)) return status::invalid_arguments;

    if (!memory) {
        *mem_object = nullptr;
        return status::success;
    }
    bool args_ok = (memory->engine()->runtime_kind() == runtime_kind::ocl);
    if (!args_ok) return status::invalid_arguments;

    void *handle;
    status_t status = memory->get_data_handle(&handle);
    if (status == status::success) *mem_object = static_cast<cl_mem>(handle);

    return status;
}

status_t dnnl_ocl_interop_memory_set_mem_object(
        memory_t *memory, cl_mem mem_object) {
    bool args_ok = (memory->engine()->runtime_kind() == runtime_kind::ocl);
    if (!args_ok) return status::invalid_arguments;

    return memory->set_data_handle(static_cast<void *>(mem_object), nullptr);
}

status_t dnnl_ocl_interop_memory_get_memory_kind(
        const memory_t *memory, memory_kind_t *memory_kind) {

    bool ok = !utils::any_null(memory, memory_kind)
            && memory->engine()->runtime_kind() == runtime_kind::ocl;
    if (!ok) return status::invalid_arguments;

    *memory_kind = utils::downcast<const ocl_memory_storage_base_t *>(
            memory->memory_storage())
                           ->memory_kind();
    return status::success;
}
