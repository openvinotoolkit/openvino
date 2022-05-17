/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain src copy of the License at
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

#define BCAST 1
#define NO_BCAST 8

#define CASE(ndims, tag) \
    case ndims: return memory::format_tag::tag;

namespace dnnl {

memory::format_tag plain_format_tag(size_t ndims) {
    assert(ndims <= 12);
    switch (ndims) {
        CASE(1, a)
        CASE(2, ab)
        CASE(3, abc)
        CASE(4, abcd)
        CASE(5, abcde)
        CASE(6, abcdef)
        CASE(7, abcdefg)
        CASE(8, abcdefgh)
        CASE(9, abcdefghi)
        CASE(10, abcdefghij)
        CASE(11, abcdefghijk)
        CASE(12, abcdefghijkl)
        default: return memory::format_tag::any;
    }
}

struct binary_bcast_test_t
    : public ::testing::TestWithParam<
              std::tuple<engine::kind, memory::dims, bool>> {};

HANDLE_EXCEPTIONS_FOR_TEST_P(
        binary_bcast_test_t, TestBinaryOptimizedBroadcast) {
    auto engine_kind = std::get<0>(GetParam());
    SKIP_IF(engine_kind != get_test_engine_kind(),
            "Test prepared for a different engine kind");
    SKIP_IF(!IMPLICATION(engine_kind == engine::kind::cpu, DNNL_X64),
            "Binary impl_info_str should be same only on x64 CPU");
    engine e {engine_kind, 0};

    const auto &src1_bcast_dims = std::get<1>(GetParam());
    const size_t ndims = src1_bcast_dims.size();

    constexpr auto defualt_dt = memory::data_type::f32;
    const auto default_format = plain_format_tag(ndims);
    memory::dims default_dims;
    for (size_t d = 0; d < ndims; d++)
        default_dims.push_back(NO_BCAST);

    std::string impl_info_no_bcast, impl_info_bcast;

    auto src0_md = memory::desc(default_dims, defualt_dt, default_format, true);
    auto src1_md = memory::desc(default_dims, defualt_dt, default_format, true);
    auto dst_md = memory::desc(default_dims, defualt_dt, default_format, true);

    auto binary_d
            = binary::desc(algorithm::binary_add, src0_md, src1_md, dst_md);

    auto binary_pd = binary::primitive_desc(binary_d, e);
    ASSERT_NO_THROW(impl_info_no_bcast = binary_pd.impl_info_str(););

    memory::desc src1_bcast_md(
            src1_bcast_dims, defualt_dt, default_format, true);

    binary_d = binary::desc(
            algorithm::binary_add, src0_md, src1_bcast_md, dst_md);
    binary_pd = binary::primitive_desc(binary_d, e);

    ASSERT_NO_THROW(impl_info_bcast = binary_pd.impl_info_str(););

    const auto expect_jit = std::get<2>(GetParam());
    if (expect_jit)
        ASSERT_EQ(impl_info_no_bcast, impl_info_bcast);
    else
        ASSERT_NE(impl_info_no_bcast, impl_info_bcast);
}

INSTANTIATE_TEST_SUITE_P(CPUOptimizedDims, binary_bcast_test_t,
        ::testing::Values(
                // 5d cases
                std::make_tuple(engine::kind::cpu,
                        memory::dims {NO_BCAST, NO_BCAST, BCAST, BCAST, BCAST},
                        true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {BCAST, NO_BCAST, BCAST, BCAST, BCAST},
                        true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {
                                NO_BCAST, BCAST, NO_BCAST, NO_BCAST, NO_BCAST},
                        true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {
                                NO_BCAST, BCAST, BCAST, NO_BCAST, NO_BCAST},
                        true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {NO_BCAST, BCAST, BCAST, BCAST, NO_BCAST},
                        true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {
                                BCAST, BCAST, NO_BCAST, NO_BCAST, NO_BCAST},
                        true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {BCAST, BCAST, BCAST, NO_BCAST, NO_BCAST},
                        true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {BCAST, BCAST, BCAST, BCAST, NO_BCAST},
                        true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {BCAST, BCAST, BCAST, BCAST, BCAST}, true),
                // 4d cases
                std::make_tuple(engine::kind::cpu,
                        memory::dims {NO_BCAST, NO_BCAST, BCAST, BCAST}, true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {BCAST, NO_BCAST, BCAST, BCAST}, true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {NO_BCAST, BCAST, NO_BCAST, NO_BCAST},
                        true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {NO_BCAST, BCAST, BCAST, NO_BCAST}, true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {BCAST, BCAST, NO_BCAST, NO_BCAST}, true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {BCAST, BCAST, BCAST, NO_BCAST}, true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {BCAST, BCAST, BCAST, BCAST}, true),
                // 3d cases
                std::make_tuple(engine::kind::cpu,
                        memory::dims {NO_BCAST, NO_BCAST, BCAST}, true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {BCAST, NO_BCAST, BCAST}, true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {NO_BCAST, BCAST, NO_BCAST}, true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {BCAST, BCAST, NO_BCAST}, true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {BCAST, BCAST, BCAST}, true),
                // 2d cases
                std::make_tuple(
                        engine::kind::cpu, memory::dims {BCAST, BCAST}, true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {NO_BCAST, BCAST}, true),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {BCAST, NO_BCAST}, true),
                // 1d case
                std::make_tuple(
                        engine::kind::cpu, memory::dims {BCAST}, true)));

INSTANTIATE_TEST_SUITE_P(CPUNotOptimizedDims, binary_bcast_test_t,
        ::testing::Values(
                // selected unoptimized cases
                std::make_tuple(engine::kind::cpu,
                        memory::dims {BCAST, BCAST, NO_BCAST, BCAST, NO_BCAST},
                        false),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {BCAST, NO_BCAST, BCAST, BCAST, NO_BCAST},
                        false),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {NO_BCAST, BCAST, BCAST, BCAST, BCAST},
                        false),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {BCAST, BCAST, NO_BCAST, BCAST}, false),
                std::make_tuple(engine::kind::cpu,
                        memory::dims {NO_BCAST, BCAST, BCAST}, false)));

} // namespace dnnl
