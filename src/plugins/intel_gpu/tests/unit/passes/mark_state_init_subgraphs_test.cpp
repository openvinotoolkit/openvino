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

class mark_state_init_subgraphs_test: public ::testing::Test {
public:
    static bool check_subgraph(const program_node& node, program& program, std::vector<primitive_id> expected_subgraph) {
        const auto& variable_id = node.as<read_value>().get_primitive()->variable_id;
        if (!program.contains_state(variable_id))
            return false;

        auto& state_initializers = program.get_initializers(variable_id);
        if (state_initializers.size() != expected_subgraph.size())
            return false;

        for (auto& pid : expected_subgraph) {
            if (std::find(state_initializers.begin(), state_initializers.end(), pid) == state_initializers.end())
                return false;
        }
        return true;
    }

    void test_cross_attn_key_state_init_subgraphs(bool is_caching_test) {
        auto& engine = get_test_engine();
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));

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

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        auto prog = network->get_program();
        ASSERT_NE(prog, nullptr);

        ASSERT_TRUE(check_subgraph(prog->get_node("read_value"), *prog, {"transpose", "reshape", "fc", "input_k", "convert"}));
    }

    void test_cross_attn_value_state_init_subgraphs(bool is_caching_test) {
        auto& engine = get_test_engine();
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));

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

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        auto prog = network->get_program();
        ASSERT_NE(prog, nullptr);

        ASSERT_TRUE(check_subgraph(prog->get_node("read_value"), *prog, {"transpose", "reshape1", "fc", "input_v", "convert", "add_data"}));
    }

    void test_cross_attn_multiple_state_init_subgraphs(bool is_caching_test) {
        auto& engine = get_test_engine();
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));

        auto param_layout_dynamic = layout{ov::PartialShape{-1, -1, 512}, data_types::f16, format::bfyx};
        auto input_q_layout_dynamic = layout{ov::PartialShape{-1, 8, -1, 64}, data_types::f16, format::bfyx};
        auto weights = engine.allocate_memory({ ov::PartialShape{512, 512}, data_types::f32, format::bfyx });
        auto add_data = engine.allocate_memory({ ov::PartialShape{1, 1, 512}, data_types::f16, format::bfyx });
        auto kv_layout = layout{ov::PartialShape{-1, 8, -1, 64},  data_types::f16, format::bfyx};
        activation_additional_params act_params = {-65504, 65504};

        topology topology;
        topology.add(input_layout("param", param_layout_dynamic));
        topology.add(input_layout("input_q", input_q_layout_dynamic));
        topology.add(data("weights_k_proj", weights));
        topology.add(data("weights_v_proj", weights));
        topology.add(data("add_data", add_data));
        topology.add(reorder("convert_k_proj",
                             input_info("weights_k_proj"),
                             format::any,
                             data_types::f16,
                             std::vector<float>(),
                             reorder_mean_mode::subtract,
                             padding(),
                             true));
        topology.add(reorder("convert_v_proj",
                             input_info("weights_v_proj"),
                             format::any,
                             data_types::f16,
                             std::vector<float>(),
                             reorder_mean_mode::subtract,
                             padding(),
                             true));
        topology.add(fully_connected("fc_k_proj", input_info("param"), { "convert_k_proj" }, "", data_types::f16, 3, 2));
        topology.add(reshape("reshape_k_proj",
                             input_info("fc_k_proj"),
                             true,
                             {0, 0, 8, 64},
                             ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 8, 64},
                             reshape::reshape_mode::base));
        topology.add(permute("transpose_k_proj", input_info("reshape_k_proj"), {0, 2, 1, 3}));
        topology.add(read_value("read_value_1", {input_info("transpose_k_proj")}, "v1", {kv_layout}, data_types::f32));
        topology.add(gemm("gemm_k_proj", {input_info("input_q"), input_info("read_value_1")}, data_types::f16, {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 1, 2, 3}, 1.0f, 0.0f));
        topology.add(activation("clamp", input_info("gemm_k_proj"), activation_func::clamp, act_params));
        topology.add(fully_connected("fc_v_proj", input_info("param"), { "convert_v_proj" }, "", data_types::f16, 3, 2));
        topology.add(eltwise("add_v_proj",
                             {input_info("fc_v_proj"), input_info("add_data")},
                             eltwise_mode::sum,
                             std::vector<float>{},
                             data_types::f16,
                             ov::op::AutoBroadcastType::NUMPY,
                             true));
        topology.add(reshape("reshape_v_proj",
                             input_info("add_v_proj"),
                             true,
                             {0, 0, 8, 64},
                             ov::PartialShape{-1, -1, 8, 64},
                             reshape::reshape_mode::base));
        topology.add(permute("transpose_v_proj", input_info("reshape_v_proj"), {0, 2, 1, 3}));
        topology.add(read_value("read_value_2", {input_info("transpose_v_proj")}, "v2", {kv_layout}, data_types::f32));
        topology.add(gemm("gemm_qkv", {input_info("clamp"), input_info("read_value_2")}, data_types::f16, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 2, 1, 3}, 1.0f, 0.0f));
        topology.add(reshape("reshape_qkv",
                             input_info("gemm_qkv"),
                             true,
                             {0, 0, 512},
                             ov::PartialShape{-1, -1, 512},
                             reshape::reshape_mode::base));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        auto prog = network->get_program();
        ASSERT_NE(prog, nullptr);

        ASSERT_TRUE(check_subgraph(prog->get_node("read_value_1"), *prog, {"transpose_k_proj", "reshape_k_proj", "fc_k_proj", "convert_k_proj"}));
        ASSERT_TRUE(check_subgraph(prog->get_node("read_value_2"), *prog, {"transpose_v_proj", "reshape_v_proj", "fc_v_proj", "convert_v_proj", "add_data"}));
    }
};

TEST_F(mark_state_init_subgraphs_test, cross_attn_key_state_init_subgraphs) {
    this->test_cross_attn_key_state_init_subgraphs(false);
}

TEST_F(mark_state_init_subgraphs_test, cross_attn_value_state_init_subgraphs) {
    this->test_cross_attn_value_state_init_subgraphs(false);
}

TEST_F(mark_state_init_subgraphs_test, cross_attn_multiple_state_init_subgraphs) {
    this->test_cross_attn_multiple_state_init_subgraphs(false);
}

TEST_F(mark_state_init_subgraphs_test, cross_attn_key_state_init_subgraphs_cached) {
    this->test_cross_attn_key_state_init_subgraphs(true);
}

TEST_F(mark_state_init_subgraphs_test, cross_attn_value_state_init_subgraphs_cached) {
    this->test_cross_attn_value_state_init_subgraphs(true);
}

TEST_F(mark_state_init_subgraphs_test, cross_attn_multiple_state_init_subgraphs_cached) {
    this->test_cross_attn_multiple_state_init_subgraphs(true);
}
