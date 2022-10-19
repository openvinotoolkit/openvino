// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "program_wrapper.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/range.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "range_inst.h"


using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct range_test_params {
    layout start_layout;
    std::vector<float> start_data;
    layout stop_layout;
    std::vector<float> stop_data;
    layout step_layout;
    std::vector<float> step_data;
    data_types output_type;
    layout expected_layout;
};

class range_test : public testing::TestWithParam<range_test_params> { };

TEST_P(range_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto start_prim = std::make_shared<input_layout>("start", p.start_layout);
    auto stop_prim = std::make_shared<input_layout>("stop", p.stop_layout);
    auto step_prim = std::make_shared<input_layout>("step", p.step_layout);
    auto range_prim = std::make_shared<range>("output", std::vector<primitive_id>{ "start", "stop", "step" }, p.output_type);

    cldnn::program prog(engine);

    auto& start_node = prog.get_or_create(start_prim);
    auto& stop_node = prog.get_or_create(stop_prim);
    auto& step_node = prog.get_or_create(step_prim);
    auto& range_node = prog.get_or_create(range_prim);

    program_wrapper::add_connection(prog, start_node, range_node);
    program_wrapper::add_connection(prog, stop_node, range_node);
    program_wrapper::add_connection(prog, step_node, range_node);

    auto params = range_node.get_kernel_impl_params();

    if (p.start_layout.is_static() && p.stop_layout.is_static() && p.step_layout.is_static()) {
        auto start_mem = engine.allocate_memory(p.start_layout);
        auto stop_mem = engine.allocate_memory(p.stop_layout);
        auto step_mem = engine.allocate_memory(p.step_layout);

        set_values(start_mem, p.start_data);
        set_values(stop_mem, p.stop_data);
        set_values(step_mem, p.step_data);

        params->memory_deps = {{0, start_mem}, {1, stop_mem}, {2, step_mem}};
    }
    auto res = range_inst::calc_output_layouts<ov::PartialShape>(range_node, *params);

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, range_test,
    testing::ValuesIn(std::vector<range_test_params>{
        {
            layout{ov::PartialShape{}, data_types::f32, format::bfyx}, {2},
            layout{ov::PartialShape{}, data_types::f32, format::bfyx}, {23},
            layout{ov::PartialShape{}, data_types::f32, format::bfyx}, {3},
            data_types::i32,
            layout{ov::PartialShape{7}, data_types::i32, format::bfyx}
        },
        {
            layout{ov::PartialShape{}, data_types::f32, format::bfyx}, {23},
            layout{ov::PartialShape{}, data_types::f32, format::bfyx}, {2},
            layout{ov::PartialShape{}, data_types::f32, format::bfyx}, {-3},
            data_types::i64,
            layout{ov::PartialShape{7}, data_types::i64, format::bfyx}
        },
        {
            layout{ov::PartialShape{}, data_types::f32, format::bfyx}, {1.0f},
            layout{ov::PartialShape{}, data_types::f32, format::bfyx}, {2.5f},
            layout{ov::PartialShape{}, data_types::f32, format::bfyx}, {0.5f},
            data_types::f32,
            layout{ov::PartialShape{3}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(1), data_types::f32, format::bfyx}, {},
            layout{ov::PartialShape::dynamic(1), data_types::f32, format::bfyx}, {},
            layout{ov::PartialShape::dynamic(1), data_types::f32, format::bfyx}, {},
            data_types::i32,
            layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx}
        }
    }));

}  // shape_infer_tests
