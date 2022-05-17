/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2021 Alanna Tempest
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

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

TEST(TestsbinaryStride, StrideZero) {
    engine eng(engine::kind::cpu, 0);
    auto strm = make_stream(eng);

    auto dims = memory::dims {2, 3};
    std::vector<float> lhs = {0, 0, 0, 0, 0, 0}; // [[0 0 0][0 0 0]]
    std::vector<float> rhs = {1, 2, 3, 4, 5, 6}; // [[1 2 3][4 5 6]]
    std::vector<float> res = {0, 0, 0, 0, 0, 0};
    std::vector<float> expected_result = {1, 1, 1, 4, 4, 4};

    const auto dst_md = memory::desc(
            dims, memory::data_type::f32, memory::format_tag::ab);

    const auto src0_md
            = memory::desc(dims, memory::data_type::f32, memory::dims {3, 1});
    const auto src1_md
            = memory::desc(dims, memory::data_type::f32, memory::dims {3, 0});
    const auto src0_mem = memory(src0_md, eng, lhs.data());
    const auto src1_mem = memory(src1_md, eng, rhs.data());
    const auto dst_mem = memory(dst_md, eng, res.data());

    const auto binary_add_desc
            = binary::desc(algorithm::binary_add, src0_md, src1_md, dst_md);
    const auto pd = binary::primitive_desc(binary_add_desc, eng);

    const auto prim = binary(pd);
    prim.execute(strm,
            {{DNNL_ARG_SRC_0, src0_mem}, {DNNL_ARG_SRC_1, src1_mem},
                    {DNNL_ARG_DST, dst_mem}});
    strm.wait();

    auto correct_result = [&]() {
        size_t nelems = res.size();
        for (size_t i = 0; i < nelems; i++)
            if (res.at(i) != expected_result.at(i)) return false;
        return true;
    };

    ASSERT_TRUE(correct_result());
}

} // namespace dnnl
