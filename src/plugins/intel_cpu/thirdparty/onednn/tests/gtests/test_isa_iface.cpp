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

#include "tests/test_isa_common.hpp"

#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "src/cpu/platform.hpp"

namespace dnnl {

class isa_set_once_test_t : public ::testing::Test {};
TEST(isa_set_once_test, TestISASetOnce) {
    auto st = set_max_cpu_isa(cpu_isa::sse41);
    ASSERT_TRUE(st == status::success || st == status::unimplemented);
    ASSERT_TRUE(mayiuse(cpu_isa::sse41));
    st = set_max_cpu_isa(cpu_isa::sse41);
    ASSERT_TRUE(st == status::invalid_arguments || st == status::unimplemented);
};

TEST(isa_set_once_test, TestISAHintsSetOnce) {
    auto st = set_cpu_isa_hints(cpu_isa_hints::prefer_ymm);
    const bool unimplemented = st == status::unimplemented;
    ASSERT_TRUE(unimplemented || st == status::success);

    // mayiuse should disable further changes in CPU ISA hints
    ASSERT_TRUE(mayiuse(cpu_isa::sse41));
    ASSERT_TRUE(unimplemented || impl::cpu::platform::prefer_ymm_requested());

    st = set_cpu_isa_hints(cpu_isa_hints::no_hints);
    ASSERT_TRUE(unimplemented || st == status::runtime_error);
};
} // namespace dnnl
