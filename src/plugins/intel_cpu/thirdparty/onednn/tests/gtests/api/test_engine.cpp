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

#include <thread>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

class engine_test_t : public ::testing::TestWithParam<engine::kind> {
protected:
    void SetUp() override {}
    engine::kind eng_kind;
};

HANDLE_EXCEPTIONS_FOR_TEST_P(engine_test_t, TestMultithreading) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
    if (eng_kind == engine::kind::cpu) {
        EXPECT_EQ((int)engine::get_count(eng_kind), 0);
        EXPECT_ANY_THROW(engine eng(eng_kind, 0));
        return;
    }
#endif

    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine is not found.");
    engine eng {eng_kind, 0};

    memory::dims tz = {1};
    memory::desc mem_d(tz, memory::data_type::f32, memory::format_tag::x);
    auto mem = test::make_memory(mem_d, eng);

    {
        auto *ptr = mem.map_data<float>();
        GTEST_EXPECT_NE(ptr, nullptr);
        for (size_t i = 0; i < mem_d.get_size() / sizeof(float); ++i)
            ptr[i] = float(i) * (i % 2 == 0 ? 1 : -1);
        mem.unmap_data(ptr);
    }

    auto eltwise_d = eltwise_forward::desc(
            prop_kind::forward, algorithm::eltwise_relu, mem_d, 0.0f);

    std::unique_ptr<eltwise_forward> eltwise;
    std::thread create([&]() {
        eltwise.reset(new eltwise_forward({eltwise_d, eng}));
    });

    create.join();

    stream s(eng);
    std::thread exe([&]() {
        eltwise->execute(s, {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}});
        s.wait();
    });

    exe.join();
}

INSTANTIATE_TEST_SUITE_P(AllEngineKinds, engine_test_t,
        ::testing::Values(engine::kind::cpu, engine::kind::gpu));

} // namespace dnnl
