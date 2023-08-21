// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reverse_sequence.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "reverse_sequence_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct reverse_sequence_test_params {
    layout input_layout;
    int32_t seq_axis;
    int32_t batch_axis;
};

class reverse_sequence_test : public testing::TestWithParam<reverse_sequence_test_params> { };

TEST_P(reverse_sequence_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input1_layout = layout{{p.input_layout.get_partial_shape()[p.batch_axis]}, data_types::i32, format::bfyx};
    auto input0_layout_prim = std::make_shared<input_layout>("input0", p.input_layout);
    auto input1_layout_prim = std::make_shared<input_layout>("input1", input1_layout);
    auto reverse_sequence_prim = std::make_shared<reverse_sequence>("output", input_info("input0"), input_info("input1"),
                                                                    p.seq_axis, p.batch_axis);

    cldnn::program prog(engine);

    auto& input0_layout_node = prog.get_or_create(input0_layout_prim);
    auto& input1_layout_node = prog.get_or_create(input1_layout_prim);
    auto& reverse_sequence_node = prog.get_or_create(reverse_sequence_prim);
    program_wrapper::add_connection(prog, input0_layout_node, reverse_sequence_node);
    program_wrapper::add_connection(prog, input1_layout_node, reverse_sequence_node);
    auto res = reverse_sequence_inst::calc_output_layouts<ov::PartialShape>(reverse_sequence_node, *reverse_sequence_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    auto expected_layout = p.input_layout;
    ASSERT_EQ(res[0], expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, reverse_sequence_test,
    testing::ValuesIn(std::vector<reverse_sequence_test_params>{
        { layout{ov::PartialShape{1, 2, 3}, data_types::f32, format::bfyx}, 1, 0},
        { layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx}, 1, 0},
        { layout{ov::PartialShape{1, 2, 3, 4, 5}, data_types::f32, format::bfzyx}, 1, 0},
        { layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}, 1, 0},
        { layout{ov::PartialShape::dynamic(5), data_types::f32, format::bfzyx}, 1, 0}
    }));

}  // shape_infer_tests
