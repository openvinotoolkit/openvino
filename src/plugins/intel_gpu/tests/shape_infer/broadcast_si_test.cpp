// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/broadcast.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "broadcast_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct broadcast_test_params {
    layout data_layout;
    layout target_shape_layout;
    ov::Shape target_shape_data;
    ngraph::AxisSet axes_mapping_data;
    ov::op::BroadcastModeSpec mode;
    layout expected_layout;
};

class broadcast_test_two_inputs : public testing::TestWithParam<broadcast_test_params> { };

TEST_P(broadcast_test_two_inputs, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto data_layout_prim = std::make_shared<input_layout>("data", p.data_layout);
    auto target_shape_layout_prim = std::make_shared<input_layout>("target_shape", p.target_shape_layout);
    auto broadcast_prim = std::make_shared<broadcast>("output", input_info("data"), input_info("target_shape"), p.axes_mapping_data, p.mode);

    cldnn::program prog(engine);

    auto target_shape_mem = engine.allocate_memory(p.target_shape_layout);
    set_values(target_shape_mem, p.target_shape_data);

    auto& data_node = prog.get_or_create(data_layout_prim);
    auto& target_shape_node = prog.get_or_create(target_shape_layout_prim);
    auto& broadcast_node = prog.get_or_create(broadcast_prim);
    program_wrapper::add_connection(prog, data_node, broadcast_node);
    program_wrapper::add_connection(prog, target_shape_node, broadcast_node);

    auto params = broadcast_node.get_kernel_impl_params();
    params->memory_deps = {{1, target_shape_mem}};
    auto res = broadcast_inst::calc_output_layouts<ov::PartialShape>(broadcast_node, *params);

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, broadcast_test_two_inputs,
    testing::ValuesIn(std::vector<broadcast_test_params>{
        {
            layout{ov::PartialShape{16, 1, 1, 1}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{5}, data_types::i64, format::bfzyx}, {1, 16, 50, 50, 50},
            {}, ov::op::BroadcastType::NUMPY,
            layout{ov::PartialShape{1, 16, 50, 50, 50}, data_types::f32, format::bfzyx}
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            layout{ov::PartialShape{5}, data_types::i64, format::bfzyx}, {1, 16, 50, 50, 50},
            {}, ov::op::BroadcastType::NUMPY,
            layout{ov::PartialShape{1, 16, 50, 50, 50}, data_types::f32, format::bfzyx}
        },
        {
            layout{ov::PartialShape{16}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{4}, data_types::i64, format::bfyx}, {1, 16, 50, 50},
            {1}, ov::op::BroadcastType::EXPLICIT,
            layout{ov::PartialShape{1, 16, 50, 50}, data_types::f32, format::bfyx}
        }
    }));

class broadcast_test_single_input : public testing::TestWithParam<broadcast_test_params> { };

TEST_P(broadcast_test_single_input, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto data_layout_prim = std::make_shared<input_layout>("data", p.data_layout);
    auto broadcast_prim = std::make_shared<broadcast>("output", input_info("data"), p.target_shape_data, p.axes_mapping_data, p.mode);

    cldnn::program prog(engine);

    auto& data_node = prog.get_or_create(data_layout_prim);
    auto& broadcast_node = prog.get_or_create(broadcast_prim);
    program_wrapper::add_connection(prog, data_node, broadcast_node);

    auto params = broadcast_node.get_kernel_impl_params();
    auto res = broadcast_inst::calc_output_layouts<ov::PartialShape>(broadcast_node, *params);

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, broadcast_test_single_input,
    testing::ValuesIn(std::vector<broadcast_test_params>{
        {
            layout{ov::PartialShape{16, 1, 1, 1}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{5}, data_types::i64, format::bfzyx}, {1, 16, 50, 50, 50},
            {}, ov::op::BroadcastType::NUMPY,
            layout{ov::PartialShape{1, 16, 50, 50, 50}, data_types::f32, format::bfzyx}
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            layout{ov::PartialShape{5}, data_types::i64, format::bfzyx}, {1, 16, 50, 50, 50},
            {}, ov::op::BroadcastType::NUMPY,
            layout{ov::PartialShape{1, 16, 50, 50, 50}, data_types::f32, format::bfzyx}
        },
        {
            layout{ov::PartialShape{16}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{4}, data_types::i64, format::bfyx}, {1, 16, 50, 50},
            {1}, ov::op::BroadcastType::EXPLICIT,
            layout{ov::PartialShape{1, 16, 50, 50}, data_types::f32, format::bfyx}
        }
    }));

}  // shape_infer_tests
