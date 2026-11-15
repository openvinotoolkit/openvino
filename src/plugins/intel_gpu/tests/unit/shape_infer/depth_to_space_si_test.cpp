// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/depth_to_space.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include "depth_to_space_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct depth_to_space_test_params {
    layout input_layout;
    size_t block_size;
    layout expected_layout;
};

class depth_to_space_test : public testing::TestWithParam<depth_to_space_test_params> { };

TEST_P(depth_to_space_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input_layout_prim = std::make_shared<input_layout>("input", p.input_layout);
    auto depth_to_space_prim = std::make_shared<depth_to_space>("depth_to_space", input_info("input"), p.block_size, depth_to_space_mode::blocks_first);

    cldnn::program prog(engine);

    auto& input_layout_node = prog.get_or_create(input_layout_prim);
    auto& depth_to_space_node = prog.get_or_create(depth_to_space_prim);
    program_wrapper::add_connection(prog, input_layout_node, depth_to_space_node);
    auto res = depth_to_space_inst::calc_output_layouts<ov::PartialShape>(depth_to_space_node, *depth_to_space_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

TEST(depth_to_space_test, grouped_fused_output_type) {
    auto& engine = get_test_engine();

    const auto input_layout_desc = layout{ov::PartialShape{1, 4, 1, 1, 1}, data_types::f32, format::bfzyx};
    const auto output_shape = ov::PartialShape{1, 2, 1, 2, 2};
    auto input_layout_prim = std::make_shared<input_layout>("input", input_layout_desc);
    auto depth_to_space_prim = std::make_shared<depth_to_space>("depth_to_space", input_info("input"), 2, 2, 2, 1);
    auto reorder_prim = std::make_shared<reorder>("reorder", input_info("depth_to_space"), format::bfzyx, data_types::i8);

    cldnn::program prog(engine);
    auto& input_layout_node = prog.get_or_create(input_layout_prim);
    auto& depth_to_space_node = prog.get_or_create(depth_to_space_prim);
    program_wrapper::add_connection(prog, input_layout_node, depth_to_space_node);

    auto impl_params = depth_to_space_node.get_kernel_impl_params();
    fused_primitive_desc fused_reorder(reorder_prim);
    fused_reorder.output_layouts = {layout{output_shape, data_types::i8, format::bfzyx}};
    impl_params->fused_desc = {fused_reorder};

    const auto result = depth_to_space_inst::calc_output_layout(depth_to_space_node, *impl_params);
    EXPECT_EQ(result, (layout{output_shape, data_types::i8, format::bfzyx}));
}

INSTANTIATE_TEST_SUITE_P(smoke, depth_to_space_test,
    testing::ValuesIn(std::vector<depth_to_space_test_params>{
        {
            layout{ov::PartialShape{5, 28, 2, 3}, data_types::f32, format::bfyx},
            2,
            layout{ov::PartialShape{5, 7, 4, 6}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            2,
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}
        }
    }));

}  // shape_infer_tests
