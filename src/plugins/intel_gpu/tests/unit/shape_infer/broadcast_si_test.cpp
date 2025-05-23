// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/broadcast.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
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
    ov::AxisSet axes_mapping_data;
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

class broadcast_test_two_inputs_blocked_format : public testing::TestWithParam<broadcast_test_params> { };
TEST_P(broadcast_test_two_inputs_blocked_format, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto data_mem = engine.allocate_memory(p.data_layout);
    auto in1_mem = engine.allocate_memory(p.target_shape_layout);
    auto in2_mem = engine.allocate_memory(p.target_shape_layout);

    // data ------------|
    // shape1 (blocked)- eltwise (plain)-- broadcast
    // shape2 (blocked) /
    // Expectation: eltwise's result is to be used as shape_mem of broadcast, and it should be plain format
    topology topology;
    topology.add(input_layout("data", layout{ov::PartialShape::dynamic(p.data_layout.get_rank()), p.data_layout.data_type, p.data_layout.format}),
                input_layout("shape_input_1", layout{ov::PartialShape::dynamic(p.target_shape_layout.get_rank()), p.target_shape_layout.data_type, p.target_shape_layout.format}),
                input_layout("shape_input_2", layout{ov::PartialShape::dynamic(p.target_shape_layout.get_rank()), p.target_shape_layout.data_type, p.target_shape_layout.format}),
                eltwise("target_shape", input_info("shape_input_1"), input_info("shape_input_2"), eltwise_mode::sum, ov::op::AutoBroadcastType::NUMPY),
                broadcast("output", input_info("data"), input_info("target_shape"), p.axes_mapping_data, p.mode)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    std::vector<int32_t> input_data(p.data_layout.get_linear_size(), 1);

    network network(engine, topology, config);

    set_values(data_mem, input_data);
    set_values(in1_mem, p.target_shape_data);
    set_values(in2_mem, p.target_shape_data);

    network.set_input_data("data", data_mem);
    network.set_input_data("shape_input_1", in1_mem);
    network.set_input_data("shape_input_2", in2_mem);

    auto outputs = network.execute();
    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    ASSERT_EQ(output->get_layout(), p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, broadcast_test_two_inputs_blocked_format,
    testing::ValuesIn(std::vector<broadcast_test_params>{
        {
            layout{ov::PartialShape{8}, data_types::i32, format::b_fs_yx_fsv16}, //data layout
            layout{ov::PartialShape{4}, data_types::i64, format::b_fs_yx_fsv16},
            {4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0}, ov::op::BroadcastType::EXPLICIT,
            layout{ov::PartialShape{8, 64, 22, 16}, data_types::i32, format::b_fs_yx_fsv16}
        },
        {
            layout{ov::PartialShape{16, 1, 1, 1}, data_types::i32, format::b_fs_yx_fsv16}, //data layout
            layout{ov::PartialShape{4}, data_types::i64, format::b_fs_yx_fsv16},
            {8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {}, ov::op::BroadcastType::NUMPY,
            layout{ov::PartialShape{16, 50, 24, 20}, data_types::i32, format::b_fs_yx_fsv16}
        },
        {
            layout{ov::PartialShape{16}, data_types::i32, format::b_fs_zyx_fsv16}, //data layout
            layout{ov::PartialShape{5}, data_types::i64, format::b_fs_zyx_fsv16},
            {8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0}, ov::op::BroadcastType::EXPLICIT,
            layout{ov::PartialShape{16, 2, 50, 24, 20}, data_types::i32, format::b_fs_zyx_fsv16}
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
