// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/strided_slice.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "strided_slice_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct strided_slice_test_params {
    layout in_layout;
    layout begin_layout;
    std::vector<int64_t> begin_data;
    layout end_layout;
    std::vector<int64_t> end_data;
    layout strides_layout;
    std::vector<int64_t> strides_data;
    std::vector<int64_t> begin_mask;
    std::vector<int64_t> end_mask;
    std::vector<int64_t> new_axis_mask;
    std::vector<int64_t> shrink_axis_mask;
    std::vector<int64_t> ellipsis_mask;
    layout expected_layout;
};

class strided_slice_test : public testing::TestWithParam<strided_slice_test_params> { };

TEST_P(strided_slice_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input_prim = std::make_shared<input_layout>("input", p.in_layout);
    auto strided_slice_prim = std::make_shared<strided_slice>("output",
                                                              input_info("input"),
                                                              p.begin_data,
                                                              p.end_data,
                                                              p.strides_data,
                                                              p.begin_mask,
                                                              p.end_mask,
                                                              p.new_axis_mask,
                                                              p.shrink_axis_mask,
                                                              p.ellipsis_mask,
                                                              ov::Shape{});

    cldnn::program prog(engine);

    auto begin_mem = engine.allocate_memory(p.begin_layout);
    auto end_mem = engine.allocate_memory(p.end_layout);
    auto strides_mem = engine.allocate_memory(p.strides_layout);
    set_values(begin_mem, p.begin_data);
    set_values(end_mem, p.end_data);
    set_values(strides_mem, p.strides_data);

    auto& input_node = prog.get_or_create(input_prim);
    auto& strided_slice_node = prog.get_or_create(strided_slice_prim);
    program_wrapper::add_connection(prog, input_node, strided_slice_node);
    auto params = strided_slice_node.get_kernel_impl_params();
    auto res = strided_slice_inst::calc_output_layouts<ov::PartialShape>(strided_slice_node, *params);

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, strided_slice_test,
    testing::ValuesIn(std::vector<strided_slice_test_params>{
        {
            layout{ov::PartialShape{1, 128, 1024}, data_types::i64, format::bfyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx}, {0, 0, 0},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx}, {0, 1, 0},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx}, {1, 1, 1},
            {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
            layout{ov::PartialShape{1, 1, 1024}, data_types::i64, format::bfyx}
        },
    }));

class strided_slice_test_four_inputs : public testing::TestWithParam<strided_slice_test_params> { };

TEST_P(strided_slice_test_four_inputs, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input_prim = std::make_shared<input_layout>("input", p.in_layout);
    auto begin_prim = std::make_shared<input_layout>("begin", p.begin_layout);
    auto end_prim = std::make_shared<input_layout>("end", p.end_layout);
    auto strides_prim = std::make_shared<input_layout>("strides", p.strides_layout);
    auto strided_slice_prim = std::make_shared<strided_slice>("output",
                                                              input_info("input"),
                                                              input_info("begin"),
                                                              input_info("end"),
                                                              input_info("strides"),
                                                              p.begin_mask,
                                                              p.end_mask,
                                                              p.new_axis_mask,
                                                              p.shrink_axis_mask,
                                                              p.ellipsis_mask,
                                                              ov::Shape{});

    cldnn::program prog(engine);

    auto begin_mem = engine.allocate_memory(p.begin_layout);
    auto end_mem = engine.allocate_memory(p.end_layout);
    auto strides_mem = engine.allocate_memory(p.strides_layout);
    set_values(begin_mem, p.begin_data);
    set_values(end_mem, p.end_data);
    set_values(strides_mem, p.strides_data);

    auto& input_node = prog.get_or_create(input_prim);
    auto& begin_node = prog.get_or_create(begin_prim);
    auto& end_node = prog.get_or_create(end_prim);
    auto& strides_node = prog.get_or_create(strides_prim);
    auto& strided_slice_node = prog.get_or_create(strided_slice_prim);
    program_wrapper::add_connection(prog, input_node, strided_slice_node);
    program_wrapper::add_connection(prog, begin_node, strided_slice_node);
    program_wrapper::add_connection(prog, end_node, strided_slice_node);
    program_wrapper::add_connection(prog, strides_node, strided_slice_node);
    auto params = strided_slice_node.get_kernel_impl_params();
    params->memory_deps = {{1, begin_mem}, {2, end_mem}, {3, strides_mem}};
    auto res = strided_slice_inst::calc_output_layouts<ov::PartialShape>(strided_slice_node, *params);

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, strided_slice_test_four_inputs,
    testing::ValuesIn(std::vector<strided_slice_test_params>{
        {
            layout{ov::PartialShape{1, 128, 1024}, data_types::i64, format::bfyx},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx}, {0, 0, 0},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx}, {0, 1, 0},
            layout{ov::PartialShape{3}, data_types::i64, format::bfyx}, {1, 1, 1},
            {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
            layout{ov::PartialShape{1, 1, 1024}, data_types::i64, format::bfyx}
        },
        {
            layout{ov::PartialShape{200, 128}, data_types::i64, format::bfyx},
            layout{ov::PartialShape{1}, data_types::i64, format::bfyx}, {0},
            layout{ov::PartialShape{1}, data_types::i64, format::bfyx}, {15},
            layout{ov::PartialShape{1}, data_types::i64, format::bfyx}, {1},
            {0}, {0}, {}, {}, {},
            layout{ov::PartialShape{15, 128}, data_types::i64, format::bfyx}
        },
    }));

}  // shape_infer_tests
