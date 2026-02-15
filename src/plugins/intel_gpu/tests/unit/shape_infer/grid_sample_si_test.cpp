// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/grid_sample.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "grid_sample_inst.hpp"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct grid_sample_test_params {
    layout input0_layout;                       // data
    layout input1_layout;                       // grid
    bool align_corners;
    GridSampleOp::InterpolationMode mode;       // BILINEAR, BICUBIC, NEAREST
    GridSampleOp::PaddingMode padding_mode;     // ZEROS, BORDER, REFLECTION
    layout expected_layout;
};

class grid_sample_test : public testing::TestWithParam<grid_sample_test_params> { };

TEST_P(grid_sample_test, shape_infer) {
    auto p = GetParam();
    const GridSampleOp::Attributes attributes(p.align_corners, p.mode, p.padding_mode);

    auto& engine = get_test_engine();
    auto input0_prim = std::make_shared<input_layout>("data", p.input0_layout);
    auto input1_prim = std::make_shared<input_layout>("grid", p.input1_layout);

    std::vector<input_info> inputs;
    inputs.push_back(input_info("data"));
    inputs.push_back(input_info("grid"));
    auto grid_sample_prim = std::make_shared<grid_sample>("grid_sample_output", inputs, attributes);

    cldnn::program prog(engine);
    auto& input0_node = prog.get_or_create(input0_prim);
    auto& input1_node = prog.get_or_create(input1_prim);
    auto& grid_sample_node = prog.get_or_create(grid_sample_prim);

    program_wrapper::add_connection(prog, input0_node, grid_sample_node);
    program_wrapper::add_connection(prog, input1_node, grid_sample_node);

    auto res = grid_sample_inst::calc_output_layouts<ov::PartialShape>(grid_sample_node, *grid_sample_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, grid_sample_test,
    testing::ValuesIn(std::vector<grid_sample_test_params>{
        {
            layout{{1, -1, -1, -1}, data_types::f32, format::bfyx},
            layout{{1, -1, -1, 2}, data_types::f32, format::bfyx},
            false,
            GridSampleOp::InterpolationMode::BILINEAR,
            GridSampleOp::PaddingMode::ZEROS,
            layout{{1, -1, -1, -1}, data_types::f32, format::bfyx}
        },
        {
            layout{{1, 2, 3, 4}, data_types::f32, format::bfyx},
            layout{{1, 5, 6, 2}, data_types::f32, format::bfyx},
            false,
            GridSampleOp::InterpolationMode::BICUBIC,
            GridSampleOp::PaddingMode::BORDER,
            layout{{1, 2, 5, 6}, data_types::f32, format::bfyx}
        },
    }));

}  // namespace shape_infer_tests
