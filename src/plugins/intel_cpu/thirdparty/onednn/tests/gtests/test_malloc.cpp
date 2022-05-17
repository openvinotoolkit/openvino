/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifdef DNNL_ENABLE_MEM_DEBUG

#ifdef _WIN32
#include <malloc.h>
#endif

#include <atomic>
#include <cstdio>
#include <cstdlib>

#include "src/common/memory_debug.hpp"
#include "tests/gtests/test_malloc.hpp"

// Counter of failed mallocs caught during execution of
// catch_expected_failures(). It is used in combination when building with
// DNNL_ENABLE_MEM_DEBUG=ON, and is useless otherwise.
static size_t failed_malloc_count = 0;

// Index of the current failed malloc. Once a malloc whose index is beyond
// failed_malloc_count, malloc fails, and the counter is incremented. Since
// there may be allocations inside parallel regions, this index is atomic to
// avoid race conditions since all the threads will write a status.
static std::atomic<size_t> last_failed_malloc_idx(0);

// Reset the total number of memory allocations (failed_malloc_count) and the
// index of the last memory allocation (last_failed_malloc_idx). This function
// has to be called once a test has passed the test and before running a
// completely new one.
void reset_failed_malloc_counter() {
    failed_malloc_count = 0;
    last_failed_malloc_idx.store(0);
}

// Increase the total number of memory allocations (failed_malloc_count) that
// are allowed to fail using the index of the last malloc
// (last_failed_malloc_idx). This function is called before rerunning a test,
// whose exception has been handled.
void increment_failed_malloc_counter() {
    failed_malloc_count = last_failed_malloc_idx.load();
    last_failed_malloc_idx.store(0);
}

bool test_out_of_memory() {
    return true;
}

namespace dnnl {
namespace impl {

// Custom malloc that replaces the one in the library during dynamical linking.
// If a malloc call, whose index is beyond the total counted
// failed_malloc_count, the malloc will fail by returning nullptr.
void *malloc(size_t size, int alignment) {
    last_failed_malloc_idx.fetch_add(1);
    if (last_failed_malloc_idx.load() > failed_malloc_count) { return nullptr; }

    return memory_debug::malloc(size, alignment);
}
} // namespace impl
} // namespace dnnl

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include <CL/cl.h>

#include "gpu/ocl/ocl_gpu_engine.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"
#include "tests/gtests/dnnl_test_common_ocl.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

// Custom clCreateBuffer wrapper that replaces the one in the library during
// dynamical linking. If an allocation, that has not been counted by
// failed_malloc_count, has been reached the wrapper allocator will fail.
cl_mem clCreateBuffer_wrapper(cl_context context, cl_mem_flags flags,
        size_t size, void *host_ptr, cl_int *errcode_ret) {
    last_failed_malloc_idx.fetch_add(1);
    if (last_failed_malloc_idx.load() > failed_malloc_count) {
        *errcode_ret = CL_MEM_OBJECT_ALLOCATION_FAILURE;
        return nullptr;
    }

    return clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
}
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

#endif
