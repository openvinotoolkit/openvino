// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <numeric>
#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/moe_fused_compressed.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>

using namespace cldnn;
using namespace ::tests;

TEST(moe_compressed_gpu, basic_compressed_moe) {
    // MOECompressed Test Configuration:
    // - 2 experts, top_k=1
    // - hidden_size=4, inter_size=6
    // - group_size=2 (for weight compression)
    // - Input: [batch_size=1, seq_len=2, hidden_size=4]
    
    const size_t batch_size = 1;
    const size_t seq_len = 2;
    const size_t hidden_size = 1024;
    const size_t inter_size = 512;
    const size_t num_experts = 4;
    const size_t top_k = 2;
    const size_t group_size = 128;
    const size_t group_num = hidden_size / group_size;
    const size_t group_num2 = inter_size / group_size;

    auto& engine = get_test_engine();

 
    // Input 0: hidden_states [batch_size, seq_len, hidden_size]
    auto hidden_states = engine.allocate_memory({
        data_types::f16, format::bfyx, 
        { batch_size, seq_len, hidden_size, 1 } 
    });
    
    set_values(hidden_states, { 
        1.0f, 2.0f, 3.0f, 4.0f,  // token 1
        0.5f, 1.5f, 2.5f, 3.5f   // token 2
    });

    // Input 1: routing_weights [batch_size, seq_len, num_experts]
    auto routing_weights = engine.allocate_memory({ 
        data_types::f16, format::bfyx, 
        {batch_size, seq_len, num_experts, 1} 
    });
    set_values(routing_weights, { 
        1.0f, 0.0f,  // token 1: expert 0 selected
        0.0f, 1.0f   // token 2: expert 1 selected
    });

    auto create_u4_weight_tensor = [&](const std::vector<int64_t>& shape, uint8_t fill_value = 1) {
        auto tensor = engine.allocate_memory({ data_types::u4, format::bfyx, 
            { shape[0], shape[1], shape[2], shape[3] } });
    
        size_t total_elements = shape[0] * shape[1] * shape[2] * shape[3];
        size_t packed_bytes = (total_elements + 1) / 2;
    
        std::vector<uint8_t> packed_data(packed_bytes);
    
        uint8_t u4_value = fill_value & 0x0F;
    
        for (size_t i = 0; i < total_elements; i += 2) {
            uint8_t packed_byte = 0;

            packed_byte |= u4_value;

            if (i + 1 < total_elements) {
                packed_byte |= (u4_value << 4);
            }

            packed_data[i / 2] = packed_byte;
        }
    
        set_values(tensor, packed_data);
        return tensor;
    };

    // Input 3: w0_weight [num_experts, inter_size, group_num, group_size]
    auto w0_weight = create_u4_weight_tensor({ 
        num_experts, inter_size, group_num, group_size 
    }, 1);

    // Input 4: w0_scale [num_experts, inter_size, group_num, 1]
    auto w0_scale = engine.allocate_memory({ 
        data_types::f16, format::bfyx, 
        { num_experts, inter_size, group_num, 1 } 
    }); 
    std::vector<ov::float16> w0_scale_data(num_experts * inter_size * group_num, 0.1f);
    set_values(w0_scale, w0_scale_data);

    // Input 5: w0_zp [num_experts, inter_size, group_num, 1]
    auto w0_zp = create_u4_weight_tensor({ 
        num_experts, inter_size, group_num, 1 
    }, 0);

    // Input 6: w1_weight [num_experts, inter_size, group_num, group_size]
    auto w1_weight = create_u4_weight_tensor({ 
        num_experts, inter_size, group_num, group_size 
    }, 1);

    // Input 7: w1_scale [num_experts, inter_size, group_num, 1]
    auto w1_scale = engine.allocate_memory({ 
        data_types::f16, format::bfyx, 
        { num_experts, inter_size, group_num, 1 } 
    });
    std::vector<ov::float16> w1_scale_data(num_experts * inter_size * group_num, 0.1f);
    set_values(w1_scale, w1_scale_data);

    // Input 8: w1_zp [num_experts, inter_size, group_num, 1]
    auto w1_zp = create_u4_weight_tensor({ 
        num_experts, inter_size, group_num, 1 
    }, 0);

    // Input 9: w2_weight [num_experts, hidden_size, group_num, group_size]
    auto w2_weight = create_u4_weight_tensor({ 
        num_experts, hidden_size, group_num2, group_size 
    }, 1);

    // Input 10: w2_scale [num_experts, hidden_size, group_num, 1]
    auto w2_scale = engine.allocate_memory({ 
        data_types::f16, format::bfyx, 
        { num_experts, hidden_size, group_num2, 1 } 
    });
    std::vector<ov::float16> w2_scale_data(num_experts * hidden_size * group_num2, 0.1f);
    set_values(w2_scale, w2_scale_data);

    // Input 11: w2_zp [num_experts, hidden_size, group_num, 1]
    auto w2_zp = create_u4_weight_tensor({ 
        num_experts, hidden_size, group_num2, 1 
    }, 0);

    // Build topology
    topology topology;
    
    // Add input layouts
    topology.add(input_layout("hidden_states", hidden_states->get_layout()));
    topology.add(input_layout("routing_weights", routing_weights->get_layout()));
    // topology.add(input_layout("topk_indices", topk_indices->get_layout()));
    
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

    // Create MOECompressed config
    cldnn::MOEFusedCompressed::Config config;
    config.hidden_size = hidden_size;
    config.inter_size = inter_size;
    config.num_expert = num_experts;
    config.top_k = top_k;
    config.group_size = group_size;
    config.out_type = data_types::f16;

    // Create MOECompressed primitive
    auto moe_prim = moe_fused_compressed("moe_fused_compressed", 
                                  {
                                      input_info("hidden_states"),
                                      input_info("routing_weights"),
                                      input_info("w0_weight"),
                                      input_info("w0_scale"),
                                      input_info("w0_zp"),
                                      input_info("w1_weight"),
                                      input_info("w1_scale"),
                                      input_info("w1_zp"),
                                      input_info("w2_weight"),
                                      input_info("w2_scale"),
                                      input_info("w2_zp")
                                  },
                                  config);

    topology.add(moe_prim);

    // Create and execute network
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("hidden_states", hidden_states);
    network.set_input_data("routing_weights", routing_weights);
    // network.set_input_data("topk_indices", topk_indices);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "moe_fused_compressed");

    auto output_prim = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(output_prim, get_test_stream());

    // Verify output shape should be [batch_size, seq_len, hidden_size]
    auto output_layout = output_prim->get_layout();
    EXPECT_EQ(output_layout.batch(), batch_size);
    EXPECT_EQ(output_layout.feature(), seq_len);
    // EXPECT_EQ(output_layout, hidden_size);

    // Basic sanity check - output should not be all zeros
    bool has_non_zero = false;
    for (size_t i = 0; i < batch_size * seq_len * hidden_size; ++i) {
        if (std::abs(output_ptr[i]) > 1e-6f) {
            has_non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero) << "Output should contain non-zero values";
}
