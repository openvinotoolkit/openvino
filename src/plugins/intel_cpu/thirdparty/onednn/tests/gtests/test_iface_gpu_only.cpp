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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "dnnl.hpp"

namespace dnnl {

TEST(iface_gpu_only, gemm) {
    auto status = dnnl_success;
    status = dnnl_sgemm(
            'N', 'N', 1, 1, 1, 1.0f, nullptr, 1, nullptr, 1, 0.0f, nullptr, 1);
    ASSERT_EQ(status, dnnl_unimplemented);

    status = dnnl_gemm_u8s8s32('N', 'N', 'C', 1, 1, 1, 1.0f, nullptr, 1, 0,
            nullptr, 1, 0, 0.0f, nullptr, 1, nullptr);
    ASSERT_EQ(status, dnnl_unimplemented);
    status = dnnl_gemm_s8s8s32('N', 'N', 'C', 1, 1, 1, 1.0f, nullptr, 1, 0,
            nullptr, 1, 0, 0.0f, nullptr, 1, nullptr);
    ASSERT_EQ(status, dnnl_unimplemented);
}

TEST(iface_gpu_only, isa) {
    ASSERT_EQ(dnnl_set_max_cpu_isa(dnnl_cpu_isa_all), dnnl_runtime_error);
    ASSERT_EQ(dnnl_get_effective_cpu_isa(), dnnl_cpu_isa_all);
    ASSERT_EQ(
            dnnl_set_cpu_isa_hints(dnnl_cpu_isa_no_hints), dnnl_runtime_error);
    ASSERT_EQ(dnnl_get_cpu_isa_hints(), dnnl_cpu_isa_no_hints);
}

TEST(iface_gpu_only, jit) {
    ASSERT_EQ(dnnl_set_jit_profiling_flags(0), dnnl_unimplemented);
    ASSERT_EQ(dnnl_set_jit_profiling_jitdumpdir(nullptr), dnnl_unimplemented);
}

} // namespace dnnl
