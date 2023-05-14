// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/tile.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "tile_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct tile_test_params {
    layout data_layout;
    layout repeats_layout;
    std::vector<int64_t> repeats_data;
    layout expected_layout;
};

class tile_test_two_inputs : public testing::TestWithParam<tile_test_params> { };

TEST_P(tile_test_two_inputs, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto data_layout_prim = std::make_shared<input_layout>("data", p.data_layout);
    auto repeats_layout_prim = std::make_shared<input_layout>("repeats", p.repeats_layout);
    auto tile_prim = std::make_shared<tile>("output", input_info("data"), input_info("repeats"));

    cldnn::program prog(engine);

    auto repeats_mem = engine.allocate_memory(p.repeats_layout);
    set_values(repeats_mem, p.repeats_data);

    auto& data_node = prog.get_or_create(data_layout_prim);
    auto& repeats_node = prog.get_or_create(repeats_layout_prim);
    auto& tile_node = prog.get_or_create(tile_prim);
    program_wrapper::add_connection(prog, data_node, tile_node);
    program_wrapper::add_connection(prog, repeats_node, tile_node);

    auto params = tile_node.get_kernel_impl_params();
    auto res = tile_inst::calc_output_layouts<ov::PartialShape>(tile_node, *params);

    auto expected_layout_dyn = p.expected_layout;
    expected_layout_dyn.set_partial_shape(ov::PartialShape::dynamic(expected_layout_dyn.get_partial_shape().size()));
    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], expected_layout_dyn);

    params->memory_deps = {{1, repeats_mem}};
    res = tile_inst::calc_output_layouts<ov::PartialShape>(tile_node, *params);

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, tile_test_two_inputs,
    testing::ValuesIn(std::vector<tile_test_params>{
        {
            layout{ov::PartialShape{2, 3, 4}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx}, {1, 2, 3},
            layout{ov::PartialShape{2, 6, 12}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx}, {1, 2, 3},
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx}
        }
    }));

class tile_test_single_input : public testing::TestWithParam<tile_test_params> { };

TEST_P(tile_test_single_input, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto data_layout_prim = std::make_shared<input_layout>("data", p.data_layout);
    auto tile_prim = std::make_shared<tile>("output", input_info("data"), p.repeats_data);

    cldnn::program prog(engine);

    auto& data_node = prog.get_or_create(data_layout_prim);
    auto& tile_node = prog.get_or_create(tile_prim);
    program_wrapper::add_connection(prog, data_node, tile_node);

    auto res = tile_inst::calc_output_layouts<ov::PartialShape>(tile_node, *tile_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, tile_test_single_input,
    testing::ValuesIn(std::vector<tile_test_params>{
        {
            layout{ov::PartialShape{2, 3, 4}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx}, {1, 2, 3},
            layout{ov::PartialShape{2, 6, 12}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx}, {1, 2, 3},
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx}
        }
    }));

}  // shape_infer_tests
