/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <memory>
#include <cstring>

#include "gtest/gtest.h"
#include "mkldnn_test_common.hpp"

#include "mkldnn.hpp"

namespace mkldnn {

typedef float data_t;

class memory_test: public ::testing::Test {
protected:
    virtual void SetUp() {}
};

TEST_F(memory_test, DataZeroDim) {
    auto e = engine(engine::kind::cpu, 0);
    mkldnn::memory mem0({{{2, 0, 3, 4}, memory::data_type::f32,
            memory::format::nChw16c}, e});
}

TEST_F(memory_test, DataPaddingTest) {
    auto e = engine(engine::kind::cpu, 0);

    const int N = 2, C = 28, C_16 = 32, H = 3, W = 4;
    const size_t phys_sz = (size_t)N * C_16 * H * W;

    mkldnn::memory mem0({{{N, C, H, W}, memory::data_type::f32,
            memory::format::nChw16c}, e});
    data_t *mem0_ptr = (data_t *)mem0.get_data_handle();
    fill_data<data_t>(N*C_16*H*W, mem0_ptr);

    std::vector<data_t> mem1_vec(phys_sz);
    mem1_vec.assign(mem0_ptr,
            mem0_ptr + mem0.get_primitive_desc().get_size() / sizeof(data_t));

    mkldnn::memory mem1({{{N, C, H, W}, memory::data_type::f32,
            memory::format::nChw16c}, e}, &mem1_vec[0]);

    check_zero_tail<data_t>(0, mem1);
    check_zero_tail<data_t>(1, mem0);

    for (size_t i = 0; i < phys_sz; ++i)
        EXPECT_NEAR(mem0_ptr[i], mem1_vec[i], 1e-7) << i;
}

TEST_F(memory_test, WeightPaddingTest) {
    auto e = engine(engine::kind::cpu, 0);

    const int O = 13, O_16 = 16, I = 28, I_16 = 32, H = 2, W = 3;
    const size_t phys_sz = (size_t)O_16 * I_16 * H * W;

    mkldnn::memory mem0({{{O, I, H, W}, memory::data_type::f32,
            memory::format::OIhw16i16o}, e});
    data_t *mem0_ptr = (data_t *)mem0.get_data_handle();
    fill_data<data_t>(O_16*I_16*H*W, mem0_ptr);

    std::vector<data_t> mem1_vec(phys_sz);
    mem1_vec.assign(mem0_ptr,
            mem0_ptr + mem0.get_primitive_desc().get_size() / sizeof(data_t));

    mkldnn::memory mem1({{{O, I, H, W}, memory::data_type::f32,
            memory::format::OIhw16i16o}, e}, &mem1_vec[0]);

    check_zero_tail<data_t>(0, mem1);
    check_zero_tail<data_t>(1, mem0);

    for (size_t i = 0; i < phys_sz; ++i)
        EXPECT_NEAR(mem0_ptr[i], mem1_vec[i], 1e-7) << i;
}

}
