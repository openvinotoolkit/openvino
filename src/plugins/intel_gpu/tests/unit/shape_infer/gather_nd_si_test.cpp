// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gather_nd.hpp>

#include "gather_nd_inst.h"
#include "program_wrapper.h"

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct gather_nd_test_params {
    layout in0_layout;
    layout in1_layout;
    int64_t batch_dim;
    bool batch_merged_output;
    layout expected_layout;
};

class gather_nd_test : public testing::TestWithParam<gather_nd_test_params> {};

TEST_P(gather_nd_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input0_layout_prim = std::make_shared<input_layout>("input0", p.in0_layout);
    auto input1_layout_prim = std::make_shared<input_layout>("input1", p.in1_layout);
    uint8_t input_rank = static_cast<uint8_t>(p.in0_layout.get_partial_shape().size());
    uint8_t indices_rank = static_cast<uint8_t>(p.in1_layout.get_partial_shape().size());
    auto gather_nd_prim = std::make_shared<gather_nd>("output", input_info("input0"), input_info("input1"),
                                                      input_rank, indices_rank, p.batch_dim, p.batch_merged_output);
    cldnn::program prog(engine);

    auto& input0_layout_node = prog.get_or_create(input0_layout_prim);
    auto& input1_layout_node = prog.get_or_create(input1_layout_prim);
    auto& gather_nd_node = prog.get_or_create(gather_nd_prim);
    program_wrapper::add_connection(prog, input0_layout_node, gather_nd_node);
    program_wrapper::add_connection(prog, input1_layout_node, gather_nd_node);
    auto res = gather_nd_inst::calc_output_layouts<ov::PartialShape>(gather_nd_node, *gather_nd_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, gather_nd_test,
    testing::ValuesIn(std::vector<gather_nd_test_params>{
        {
            layout{ov::PartialShape{1000, 256, 10, 15}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{25, 125, 3}, data_types::f32, format::bfyx},
            0, false,
            layout{ov::PartialShape{25, 125, 15}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{30, 2, 100, 35}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{30, 2, 3, 1}, data_types::f32, format::bfyx},
            2, false,
            layout{ov::PartialShape{30, 2, 3, 35}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{30, 2, 100, 35}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{30, 2, 3, 1}, data_types::f32, format::bfyx},
            2, true,
            layout{ov::PartialShape{60, 3, 35}, data_types::f32, format::bfyx}
        }
    }));

}  // shape_infer_tests
