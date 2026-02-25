// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/moe_3gemm_fused_compressed.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <iostream>
#include <numeric>

#include "random_generator.hpp"
#include "test_utils.h"
#include "moe_3gemm_test_data.h"

using namespace cldnn;
using namespace ::tests;

TEST(moe_3gemm_compressed_gpu, moe_accuracy_test) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        return;
    }


    const size_t batch_size = 1;
    const size_t seq_len = 1;
    const size_t hidden_size = 128;
    const size_t inter_size = 128;
    const size_t num_experts = 4;
    const size_t top_k = 2;
    const size_t group_size = 128;
    const size_t group_num = hidden_size / group_size;
    const size_t group_num2 = inter_size / group_size;

    auto create_u4_tensor = [&](const std::vector<uint8_t>& values, int64_t b, int64_t f, int64_t y, int64_t x) {
        auto mem = engine.allocate_memory({data_types::u4, format::bfyx, {b, f, y, x}});
        set_values(mem, values);
        get_test_stream().finish();
        return mem;
    };

    auto create_f16_tensor = [&](const std::vector<ov::float16>& values, int64_t b, int64_t f, int64_t y, int64_t x) {
        auto mem = engine.allocate_memory({data_types::f16, format::bfyx, {b, f, y, x}});
        set_values(mem, values);
        get_test_stream().finish();
        return mem;
    };
    // Input 0: hidden_states [batch_size, seq_len, hidden_size]
    auto hidden_states = create_f16_tensor(hidden_states_data, batch_size, seq_len, hidden_size, 1);

    // Input 1: routing_weights [batch_size, seq_len, num_experts]
    auto routing_weights = create_f16_tensor(router_weights_data, batch_size, seq_len, num_experts, 1);

    // Input 3: w0_weight [num_experts, inter_size, group_num, group_size]
    auto w0_weight = create_u4_tensor(w0_weights_data, num_experts, inter_size, group_num, group_size);

    // Input 4: w0_scale [num_experts, inter_size, group_num, 1]
    auto w0_scale = create_f16_tensor(w0_scale_data, num_experts, inter_size, group_num, 1);

    // Input 5: w0_zp [num_experts, inter_size, group_num, 1]
    auto w0_zp = create_u4_tensor(w0_zp_data, num_experts, inter_size, group_num, 1);

    // Input 6: w1_weight [num_experts, inter_size, group_num, group_size]
    auto w1_weight = create_u4_tensor(w1_weights_data, num_experts, inter_size, group_num, group_size);

    // Input 7: w1_scale [num_experts, inter_size, group_num, 1]
    auto w1_scale = create_f16_tensor(w1_scale_data, num_experts, inter_size, group_num, 1);
    // Input 8: w1_zp [num_experts, inter_size, group_num, 1]
    auto w1_zp = create_u4_tensor(w1_zp_data, num_experts, inter_size, group_num, group_size);

    // Input 9: w2_weight [num_experts, hidden_size, group_num, group_size]
    auto w2_weight = create_u4_tensor(w2_weights_data, num_experts, hidden_size, group_num2, group_size);

    // Input 10: w2_scale [num_experts, hidden_size, group_num, 1]
    auto w2_scale = create_f16_tensor(w2_scale_data, num_experts, hidden_size, group_num2, 1);
    // Input 11: w2_zp [num_experts, hidden_size, group_num, 1]
    auto w2_zp = create_u4_tensor(w2_zp_data, num_experts, hidden_size, group_num2, 1);

    // Input 3: w0_weight [num_experts, inter_size, group_num, group_size]
    // Build topology
    topology topology;

    // Add input layouts
    topology.add(input_layout("hidden_states", hidden_states->get_layout()));
    topology.add(input_layout("routing_weights", routing_weights->get_layout()));

    // Add weight data
    topology.add(data("w0_weight", w0_weight));
    topology.add(data("w0_scale", w0_scale));
    topology.add(data("w0_zp", w0_zp));
    topology.add(data("w1_weight", w1_weight));
    topology.add(data("w1_scale", w1_scale));
    topology.add(data("w1_zp", w1_zp));
    topology.add(data("w2_weight", w2_weight));
    topology.add(data("w2_scale", w2_scale));
    topology.add(data("w2_zp", w2_zp));

    // Create MOE3GemmFusedCompressed config
    cldnn::MOE3GemmFusedCompressed::Config config;
    config.hidden_size = hidden_size;
    config.inter_size = inter_size;
    config.num_expert = num_experts;
    config.top_k = top_k;
    config.group_size = group_size;
    config.out_type = data_types::f16;

    // Create MOECompressed primitive
    auto moe_prim = moe_3gemm_fused_compressed("moe_3gemm_fused_compressed",
                                         {input_info("hidden_states"),
                                          input_info("routing_weights"),
                                          input_info("w0_weight"),
                                          input_info("w0_scale"),
                                          input_info("w0_zp"),
                                          input_info("w1_weight"),
                                          input_info("w1_scale"),
                                          input_info("w1_zp"),
                                          input_info("w2_weight"),
                                          input_info("w2_scale"),
                                          input_info("w2_zp")},
                                         config);

    topology.add(moe_prim);

    // Create and execute network
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("hidden_states", hidden_states);
    network.set_input_data("routing_weights", routing_weights);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "moe_3gemm_fused_compressed");

    auto output_prim = outputs.begin()->second.get_memory();
    get_test_stream().flush();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_prim, get_test_stream());

    // Verify output shape should be [batch_size, seq_len, hidden_size]
    auto output_layout = output_prim->get_layout();
    EXPECT_EQ(output_layout.batch(), batch_size);
    EXPECT_EQ(output_layout.feature(), seq_len);

    for (size_t i = 0; i < batch_size * seq_len * hidden_size; ++i) {
        EXPECT_NEAR(static_cast<float>(output_ptr[i]), static_cast<float>(output_ref[i]), 1e-3f);
    }
}