// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/border.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "border_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct pad_test_params {
    layout in_layout;
    layout pads_begin_layout;
    ov::CoordinateDiff pads_begin_data;
    layout pads_end_layout;
    ov::CoordinateDiff pads_end_data;
    ov::op::PadMode pad_mode;
    float pad_value;
    layout expected_layout;
};

class pad_test_three_input : public testing::TestWithParam<pad_test_params> { };

TEST_P(pad_test_three_input, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input_prim = std::make_shared<input_layout>("input", p.in_layout);
    auto pads_begin_prim = std::make_shared<input_layout>("pads_begin", p.pads_begin_layout);
    auto pads_end_prim = std::make_shared<input_layout>("pads_end", p.pads_end_layout);

    int32_t non_constant_input_mask = border::PAD_NON_CONST_INPUT::BEGIN | border::PAD_NON_CONST_INPUT::END;
    auto border_prim = std::make_shared<border>("output", std::vector<input_info>{input_info("input"), input_info("pads_begin"), input_info("pads_end")},
                                                non_constant_input_mask,
                                                std::vector<int64_t>{}, std::vector<int64_t>{},
                                                p.pad_mode, p.pad_value);
    cldnn::program prog(engine);

    auto pads_begin_mem = engine.allocate_memory(p.pads_begin_layout);
    auto pads_end_mem = engine.allocate_memory(p.pads_end_layout);
    set_values(pads_begin_mem, p.pads_begin_data);
    set_values(pads_end_mem, p.pads_end_data);

    auto& input_node = prog.get_or_create(input_prim);
    auto& pads_begin_node = prog.get_or_create(pads_begin_prim);
    auto& pads_end_node = prog.get_or_create(pads_end_prim);
    auto& border_node = prog.get_or_create(border_prim);
    program_wrapper::add_connection(prog, input_node, border_node);
    program_wrapper::add_connection(prog, pads_begin_node, border_node);
    program_wrapper::add_connection(prog, pads_end_node, border_node);

    auto params = border_node.get_kernel_impl_params();
    params->memory_deps = {{1, pads_begin_mem}, {2, pads_end_mem}};
    auto res = border_inst::calc_output_layouts<ov::PartialShape>(border_node, *params);

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, pad_test_three_input,
    testing::ValuesIn(std::vector<pad_test_params>{
        {
            layout{ov::PartialShape{1, 3, 32, 40}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{4}, data_types::i64, format::bfyx}, {0, 5, 2, 1},
            layout{ov::PartialShape{4}, data_types::i64, format::bfyx}, {1, 0, 3, 7},
            ov::op::PadMode::CONSTANT, 1.f,
            layout{ov::PartialShape{2, 8, 37, 48}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            layout{ov::PartialShape{4}, data_types::i64, format::bfyx}, {0, 5, 2, 1},
            layout{ov::PartialShape{4}, data_types::i64, format::bfyx}, {1, 0, 3, 7},
            ov::op::PadMode::CONSTANT, 1.f,
            layout{ov::PartialShape{{1, -1}, {5, -1}, {5, -1}, {8, -1}}, data_types::f32, format::bfyx}
        }
    }));

class pad_test_single_input : public testing::TestWithParam<pad_test_params> { };

TEST_P(pad_test_single_input, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input_prim = std::make_shared<input_layout>("input", p.in_layout);

    auto border_prim = std::make_shared<border>("output", std::vector<input_info>({input_info("input")}), 0,
                                                p.pads_begin_data, p.pads_end_data,
                                                p.pad_mode, p.pad_value);
    cldnn::program prog(engine);

    auto& input_node = prog.get_or_create(input_prim);
    auto& border_node = prog.get_or_create(border_prim);
    program_wrapper::add_connection(prog, input_node, border_node);

    auto res = border_inst::calc_output_layouts<ov::PartialShape>(border_node, *border_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, pad_test_single_input,
    testing::ValuesIn(std::vector<pad_test_params>{
        {
            layout{ov::PartialShape{1, 3, 32, 40}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{4}, data_types::i64, format::bfyx}, {0, 5, 2, 1},
            layout{ov::PartialShape{4}, data_types::i64, format::bfyx}, {1, 0, 3, 7},
            ov::op::PadMode::CONSTANT, 1.f,
            layout{ov::PartialShape{2, 8, 37, 48}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            layout{ov::PartialShape{4}, data_types::i64, format::bfyx}, {0, 5, 2, 1},
            layout{ov::PartialShape{4}, data_types::i64, format::bfyx}, {1, 0, 3, 7},
            ov::op::PadMode::CONSTANT, 1.f,
            layout{ov::PartialShape{{1, -1}, {5, -1}, {5, -1}, {8, -1}}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx},
            layout{ov::PartialShape{2}, data_types::i64, format::bfyx}, {0, 5},
            layout{ov::PartialShape{2}, data_types::i64, format::bfyx}, {1, 0},
            ov::op::PadMode::CONSTANT, 1.f,
            layout{ov::PartialShape{{1, -1}, {5, -1}}, data_types::f32, format::bfyx}
        }
    }));

}  // namespace shape_infer_tests
