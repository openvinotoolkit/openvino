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

#include "src/gpu/ocl/ocl_usm_utils.hpp"

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_ocl.hpp"

#include <cstdint>
#include <vector>

using namespace dnnl::impl::gpu::ocl;

namespace dnnl {

namespace {
void fill_data(void *usm_ptr, memory::dim n, const engine &eng) {
    auto alloc_kind = usm::get_pointer_type(eng.get(), usm_ptr);
    if (alloc_kind == usm::ocl_usm_kind_t::host
            || alloc_kind == usm::ocl_usm_kind_t::shared) {
        for (int i = 0; i < n; i++)
            ((float *)usm_ptr)[i] = float(i);
    } else {
        std::vector<float> host_ptr(n);
        for (int i = 0; i < n; i++)
            host_ptr[i] = float(i);

        auto s = stream(eng);
        usm::memcpy(s.get(), usm_ptr, host_ptr.data(), n * sizeof(float));
        s.wait();
    }
}
} // namespace

class ocl_memory_usm_test_t : public ::testing::Test {};

HANDLE_EXCEPTIONS_FOR_TEST(ocl_memory_usm_test_t, Constructor) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    void *ptr = usm::malloc_shared(eng.get(), sizeof(float) * n);

    auto mem = ocl_interop::make_memory(
            mem_d, eng, ocl_interop::memory_kind::usm, ptr);

    ASSERT_EQ(ptr, mem.get_data_handle());
    ASSERT_EQ(ocl_interop::memory_kind::usm, ocl_interop::get_memory_kind(mem));

    {
        for (int i = 0; i < n; i++) {
            ((float *)ptr)[i] = float(i);
        }
    }

    {
        float *ptr_f32 = (float *)mem.get_data_handle();
        GTEST_EXPECT_NE(ptr_f32, nullptr);
        for (int i = 0; i < n; i++) {
            ASSERT_EQ(ptr_f32[i], float(i));
        }
    }

    usm::free(eng.get(), ptr);
}

HANDLE_EXCEPTIONS_FOR_TEST(ocl_memory_usm_test_t, ConstructorNone) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);
    memory::desc mem_d({0}, memory::data_type::f32, memory::format_tag::x);

    auto mem = ocl_interop::make_memory(
            mem_d, eng, ocl_interop::memory_kind::usm, DNNL_MEMORY_NONE);

    ASSERT_EQ(nullptr, mem.get_data_handle());
    ASSERT_EQ(ocl_interop::memory_kind::usm, ocl_interop::get_memory_kind(mem));
}

HANDLE_EXCEPTIONS_FOR_TEST(ocl_memory_usm_test_t, ConstructorAllocate) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    auto mem = ocl_interop::make_memory(
            mem_d, eng, ocl_interop::memory_kind::usm, DNNL_MEMORY_ALLOCATE);

    ASSERT_EQ(ocl_interop::memory_kind::usm, ocl_interop::get_memory_kind(mem));

    void *ptr = mem.get_data_handle();
    GTEST_EXPECT_NE(ptr, nullptr);
    fill_data(ptr, n, eng);

    float *mapped_ptr = mem.map_data<float>();
    GTEST_EXPECT_NE(mapped_ptr, nullptr);
    for (int i = 0; i < n; i++) {
        ASSERT_EQ(mapped_ptr[i], float(i));
    }
    mem.unmap_data(mapped_ptr);
}

HANDLE_EXCEPTIONS_FOR_TEST(ocl_memory_usm_test_t, DefaultConstructor) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    auto mem = ocl_interop::make_memory(
            mem_d, eng, ocl_interop::memory_kind::usm);

    ASSERT_EQ(ocl_interop::memory_kind::usm, ocl_interop::get_memory_kind(mem));

    void *ptr = mem.get_data_handle();
    GTEST_EXPECT_NE(ptr, nullptr);
    fill_data(ptr, n, eng);

    float *mapped_ptr = mem.map_data<float>();
    GTEST_EXPECT_NE(mapped_ptr, nullptr);
    for (int i = 0; i < n; i++) {
        ASSERT_EQ(mapped_ptr[i], float(i));
    }
    mem.unmap_data(mapped_ptr);
}

} // namespace dnnl
