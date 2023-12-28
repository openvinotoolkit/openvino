// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/space_to_depth.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "space_to_depth_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct space_to_depth_test_params {
    layout input_layout;
    size_t block_size;
    layout expected_layout;
};

class space_to_depth_si_test : public testing::TestWithParam<space_to_depth_test_params> { };

TEST_P(space_to_depth_si_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input_layout_prim = std::make_shared<input_layout>("input", p.input_layout);
    auto space_to_depth_prim = std::make_shared<space_to_depth>("space_to_depth", input_info("input"),
                                                                        SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, p.block_size);

    cldnn::program prog(engine);
    auto& input_layout_node = prog.get_or_create(input_layout_prim);
    auto& space_to_depth_node = prog.get_or_create(space_to_depth_prim);
    program_wrapper::add_connection(prog, input_layout_node, space_to_depth_node);
    auto res = space_to_depth_inst::calc_output_layouts<ov::PartialShape>(space_to_depth_node, *space_to_depth_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, space_to_depth_si_test,
    testing::ValuesIn(std::vector<space_to_depth_test_params>{
        {
            layout{ov::PartialShape{4, 2, 4, 8}, data_types::f16, format::bfyx}, 4, layout{ov::PartialShape{4, 32, 1, 2}, data_types::f16, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}, 2, layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}
        }
    }));

}  // shape_infer_tests