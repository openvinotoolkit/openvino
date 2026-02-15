// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gtest/internal/gtest-param-util.h>
#include "impls/ocl_v2/utils/jitter.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/except.hpp"
#include "primitive_inst.h"
#include "test_utils.h"
#include "registry/registry.hpp"
#include "primitive_type_base.h"
#include <memory>

using namespace cldnn;
using namespace ::tests;
using namespace ov::intel_gpu;
using namespace ov::intel_gpu::ocl;

namespace {

struct Params {
    layout l;
    size_t shape_info_offset;
    struct ExpectedResult {
        std::string dim_b;
        std::string dim_f;
        std::string dim_v;
        std::string dim_u;
        std::string dim_w;
        std::string dim_z;
        std::string dim_y;
        std::string dim_x;

        std::string pad_l_b;
        std::string pad_l_f;
        std::string pad_l_v;
        std::string pad_l_u;
        std::string pad_l_w;
        std::string pad_l_z;
        std::string pad_l_y;
        std::string pad_l_x;

        std::string pad_u_b;
        std::string pad_u_f;
        std::string pad_u_v;
        std::string pad_u_u;
        std::string pad_u_w;
        std::string pad_u_z;
        std::string pad_u_y;
        std::string pad_u_x;

        std::string stride_b;
        std::string stride_f;
        std::string stride_v;
        std::string stride_u;
        std::string stride_w;
        std::string stride_z;
        std::string stride_y;
        std::string stride_x;

        std::string offset;
    } expected;
};

class JitterTest : public ::testing::TestWithParam<Params> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<Params> &obj) {
        std::stringstream s;
        s << obj.param.l.to_short_string() << "/" << obj.index;
        return s.str();
    }
};
TEST_P(JitterTest, data_layout) {
    auto& p = GetParam();
    LayoutJitter jitter(p.l, p.shape_info_offset);

    ASSERT_EQ(jitter.dim(ChannelName::BATCH), p.expected.dim_b);
    ASSERT_EQ(jitter.dim(ChannelName::FEATURE), p.expected.dim_f);
    ASSERT_EQ(jitter.dim(ChannelName::V), p.expected.dim_v);
    ASSERT_EQ(jitter.dim(ChannelName::U), p.expected.dim_u);
    ASSERT_EQ(jitter.dim(ChannelName::W), p.expected.dim_w);
    ASSERT_EQ(jitter.dim(ChannelName::Z), p.expected.dim_z);
    ASSERT_EQ(jitter.dim(ChannelName::Y), p.expected.dim_y);
    ASSERT_EQ(jitter.dim(ChannelName::X), p.expected.dim_x);

    ASSERT_EQ(jitter.pad_l(ChannelName::BATCH), p.expected.pad_l_b);
    ASSERT_EQ(jitter.pad_l(ChannelName::FEATURE), p.expected.pad_l_f);
    ASSERT_EQ(jitter.pad_l(ChannelName::V), p.expected.pad_l_v);
    ASSERT_EQ(jitter.pad_l(ChannelName::U), p.expected.pad_l_u);
    ASSERT_EQ(jitter.pad_l(ChannelName::W), p.expected.pad_l_w);
    ASSERT_EQ(jitter.pad_l(ChannelName::Z), p.expected.pad_l_z);
    ASSERT_EQ(jitter.pad_l(ChannelName::Y), p.expected.pad_l_y);
    ASSERT_EQ(jitter.pad_l(ChannelName::X), p.expected.pad_l_x);

    ASSERT_EQ(jitter.pad_u(ChannelName::BATCH), p.expected.pad_u_b);
    ASSERT_EQ(jitter.pad_u(ChannelName::FEATURE), p.expected.pad_u_f);
    ASSERT_EQ(jitter.pad_u(ChannelName::V), p.expected.pad_u_v);
    ASSERT_EQ(jitter.pad_u(ChannelName::U), p.expected.pad_u_u);
    ASSERT_EQ(jitter.pad_u(ChannelName::W), p.expected.pad_u_w);
    ASSERT_EQ(jitter.pad_u(ChannelName::Z), p.expected.pad_u_z);
    ASSERT_EQ(jitter.pad_u(ChannelName::Y), p.expected.pad_u_y);
    ASSERT_EQ(jitter.pad_u(ChannelName::X), p.expected.pad_u_x);

    ASSERT_EQ(jitter.stride(ChannelName::BATCH), p.expected.stride_b);
    ASSERT_EQ(jitter.stride(ChannelName::FEATURE), p.expected.stride_f);
    ASSERT_EQ(jitter.stride(ChannelName::V), p.expected.stride_v);
    ASSERT_EQ(jitter.stride(ChannelName::U), p.expected.stride_u);
    ASSERT_EQ(jitter.stride(ChannelName::W), p.expected.stride_w);
    ASSERT_EQ(jitter.stride(ChannelName::Z), p.expected.stride_z);
    ASSERT_EQ(jitter.stride(ChannelName::Y), p.expected.stride_y);
    ASSERT_EQ(jitter.stride(ChannelName::X), p.expected.stride_x);

    ASSERT_EQ(jitter.offset(), p.expected.offset);
}

std::vector<Params> get_test_params() {
    return {
        {
            layout{{2, 3, 4, 5}, ov::element::f16, format::bfyx}, 0,
            Params::ExpectedResult{
                "2", "3", "1", "1", "1", "1", "4", "5",
                "0", "0", "0", "0", "0", "0", "0", "0",
                "0", "0", "0", "0", "0", "0", "0", "0",
                "60", "20", "0", "0", "0", "0", "5", "1",
                "0"
            }
        },
        {
            layout{{2, 3, 4, 5}, ov::element::f16, format::bfyx, padding{{9, 8, 7, 6}, {10, 20, 30, 40}}}, 0,
            Params::ExpectedResult{
                "2", "3", "1", "1", "1", "1", "4", "5",
                "9", "8", "0", "0", "0", "0", "7", "6",
                "10", "20", "0", "0", "0", "0", "30", "40",
                "64821", "2091", "0", "0", "0", "0", "51", "1",
                "600480"
            }
        },
        {
            layout{{2, 3, 4, 5, 6}, ov::element::f16, format::bfzyx, padding{{9, 8, 7, 6, 5}, {0, 10, 20, 30, 40}}}, 0,
            Params::ExpectedResult{
                "2", "3", "1", "1", "1", "4", "5", "6",
                "9", "8", "0", "0", "0", "7", "6", "5",
                "0", "10", "0", "0", "0", "20", "30", "40",
                "1361241", "64821", "0", "0", "0", "2091", "51", "1",
                "12784685"
            }
        },
        {
            layout{{2, 3}, ov::element::f16, format::bfzyx, padding{{1, 2}, {3, 4}}}, 0,
            Params::ExpectedResult{
                "2", "3", "1", "1", "1", "1", "1", "1",
                "1", "2", "0", "0", "0", "0", "0", "0",
                "3", "4", "0", "0", "0", "0", "0", "0",
                "9", "1", "0", "0", "0", "0", "0", "0",
                "11"
            }
        },
        {
            layout{{2, 3, 4, 5}, ov::element::f16, format::bfyx, padding{{-1, -1, -1, -1}, {10, 20, 30, 40}, padding::DynamicDimsMask{"1111"}}}, 8,
            Params::ExpectedResult{
                "2", "3", "1", "1", "1", "1", "4", "5",
                "shape_info[16]", "shape_info[18]", "0", "0", "0", "0", "shape_info[20]", "shape_info[22]",
                "shape_info[17]", "shape_info[19]", "0", "0", "0", "0", "shape_info[21]", "shape_info[23]",
                "((((3 + shape_info[18]) + shape_info[19]) * ((4 + shape_info[20]) + shape_info[21])) * ((5 + shape_info[22]) + shape_info[23]))", // stride_b
                "(((4 + shape_info[20]) + shape_info[21]) * ((5 + shape_info[22]) + shape_info[23]))", // stride_f
                "0", "0", "0", "0", "((5 + shape_info[22]) + shape_info[23])", "1", // other strides
                "(((shape_info[22] + (((5 + shape_info[22]) + shape_info[23]) * shape_info[20])) + ((((4 + shape_info[20]) + shape_info[21]) * ((5 + shape_info[22]) + shape_info[23])) * shape_info[18])) + (((((3 + shape_info[18]) + shape_info[19]) * ((4 + shape_info[20]) + shape_info[21])) * ((5 + shape_info[22]) + shape_info[23])) * shape_info[16]))"
            }
        },
    };
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         JitterTest,
                         ::testing::ValuesIn(get_test_params()),
                         JitterTest::get_test_case_name);

}  // namespace
