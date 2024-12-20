// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "activation_inst.h"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "gemm_inst.h"
#include "fully_connected_inst.h"
#include "read_value_inst.h"
#include "reorder_inst.h"
#include "reshape_inst.h"
#include "permute_inst.h"

#include "program_wrapper.h"

using namespace cldnn;
using namespace ::tests;

static bool check_subgraph(const program_node& node, const program& program, const size_t expected_num_pids) {
    auto& pids = node.get_dependant_initializer_pids();
    if (pids.size() != expected_num_pids)
        return false;

    const auto& variable_id = node.as<read_value>().get_primitive()->variable_id;
    for (auto& pid : pids) {
        if (!program.get_node(pid).is_in_state_init_subgraph())
            return false;
        if (program.get_node(pid).get_state_init_subgraph_id().compare(variable_id) != 0)
            return false;
    }
    return true;
}

TEST(mark_state_init_subgraphs, cross_attn_key_state_init_subgraphs) {
    auto& engine = get_test_engine();
    auto input_k_layout_dynamic = layout{ov::PartialShape{-1, -1, 512}, data_types::f16, format::bfyx};
    auto input_q_layout_dynamic = layout{ov::PartialShape{-1, 8, -1, 64}, data_types::f16, format::bfyx};
    auto weights = engine.allocate_memory({ ov::PartialShape{512, 512}, data_types::f32, format::bfyx });
    ov::op::util::VariableInfo info{ov::PartialShape{-1, 8, -1, 64}, data_types::f16, "v0"};
    auto kv_layout = layout{info.data_shape, info.data_type, format::bfyx};
    activation_additional_params params = {-65504, 65504};

    topology topology;
    topology.add(input_layout("input_k", input_k_layout_dynamic));
    topology.add(input_layout("input_q", input_q_layout_dynamic));
    topology.add(data("weights", weights));
    topology.add(reorder("convert",
                         input_info("weights"),
                         format::any,
                         data_types::f16,
                         std::vector<float>(),
                         reorder_mean_mode::subtract,
                         padding(),
                         true));
    topology.add(fully_connected("fc", input_info("input_k"), { "convert" }, "", data_types::f16, 3, 2));
    topology.add(reshape("reshape",
                         input_info("fc"),
                         true,
                         {0, 0, 8, 64},
                         ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 8, 64},
                         reshape::reshape_mode::base));
    topology.add(permute("transpose", input_info("reshape"), {0, 2, 1, 3}));
    topology.add(read_value("read_value", {input_info("transpose")}, info.variable_id, {kv_layout}, data_types::f32));
    topology.add(gemm("gemm", {input_info("input_q"), input_info("read_value")}, data_types::f16, {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 1, 2, 3}, 1.0f, 0.0f));
    topology.add(activation("clamp", input_info("gemm"), activation_func::clamp, params));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);

    ASSERT_TRUE(check_subgraph(prog->get_node("read_value"), *prog, 4));
}

TEST(mark_state_init_subgraphs, cross_attn_value_state_init_subgraphs) {
    auto& engine = get_test_engine();
    auto input_v_layout_dynamic = layout{ov::PartialShape{-1, -1, 512}, data_types::f16, format::bfyx};
    auto input_qk_layout_dynamic = layout{ov::PartialShape{-1, 8, -1, -1}, data_types::f16, format::bfyx};
    auto weights = engine.allocate_memory({ ov::PartialShape{512, 512}, data_types::f32, format::bfyx });
    auto add_data = engine.allocate_memory({ ov::PartialShape{1, 1, 512}, data_types::f16, format::bfyx });
    ov::op::util::VariableInfo info{ov::PartialShape{-1, 8, -1, 64}, data_types::f16, "v1"};
    auto kv_layout = layout{info.data_shape, info.data_type, format::bfyx};

    topology topology;
    topology.add(input_layout("input_v", input_v_layout_dynamic));
    topology.add(input_layout("input_qk", input_qk_layout_dynamic));
    topology.add(data("weights", weights));
    topology.add(data("add_data", add_data));
    topology.add(reorder("convert",
                         input_info("weights"),
                         format::any,
                         data_types::f16,
                         std::vector<float>(),
                         reorder_mean_mode::subtract,
                         padding(),
                         true));
    topology.add(fully_connected("fc", input_info("input_v"), { "convert" }, "", data_types::f16, 3, 2));
    topology.add(eltwise("add",
                         {input_info("fc"), input_info("add_data")},
                         eltwise_mode::sum,
                         std::vector<float>{},
                         data_types::f16,
                         ov::op::AutoBroadcastType::NUMPY,
                         true));
    topology.add(reshape("reshape1",
                         input_info("add"),
                         true,
                         {0, 0, 8, 64},
                         ov::PartialShape{-1, -1, 8, 64},
                         reshape::reshape_mode::base));
    topology.add(permute("transpose", input_info("reshape1"), {0, 2, 1, 3}));
    topology.add(read_value("read_value", {input_info("transpose")}, info.variable_id, {kv_layout}, data_types::f32));
    topology.add(gemm("gemm", {input_info("input_qk"), input_info("read_value")}, data_types::f16, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 2, 1, 3}, 1.0f, 0.0f));
    topology.add(reshape("reshape2",
                         input_info("gemm"),
                         true,
                         {0, 0, 512},
                         ov::PartialShape{-1, -1, 512},
                         reshape::reshape_mode::base));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);

    ASSERT_TRUE(check_subgraph(prog->get_node("read_value"), *prog, 4));
}
