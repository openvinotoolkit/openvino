// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/moe_3gemm_fused_compressed.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/layout.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <iostream>
#include <numeric>

#include "../../common/random_generator.hpp"
#include "../test_utils/test_utils.h"
#include "moe_3gemm_test_data.h"

using namespace cldnn;
using namespace ::tests;

struct Moe3GemmConfig {
    size_t batch_size;
    size_t seq_len;
    size_t hidden_size;
    size_t inter_size;
    size_t num_experts;
    size_t top_k;
    size_t group_size;
    bool is_u4;
};

struct Moe3GemmReference {
    Moe3GemmConfig config;
    tests::random_generator& rg;

    Moe3GemmReference(Moe3GemmConfig config, tests::random_generator& rg) : config(config), rg(rg) {}

    std::vector<uint8_t> pack(const std::vector<uint8_t>& unpacked) {
        if (!config.is_u4)
            return unpacked;

        std::vector<uint8_t> packed;
        packed.reserve((unpacked.size() + 1) / 2);
        for (size_t i = 0; i < unpacked.size(); i += 2) {
            uint8_t val1 = unpacked[i] & 0xF;
            uint8_t val2 = (i + 1 < unpacked.size()) ? (unpacked[i + 1] & 0xF) : 0;
            packed.push_back(val1 | (val2 << 4));
        }
        return packed;
    }

    std::tuple<std::vector<uint8_t>, std::vector<ov::float16>, std::vector<uint8_t>> quantize(const std::vector<float>& data,
                                                                                              size_t expert_num,
                                                                                              size_t rows,
                                                                                              size_t cols,
                                                                                              size_t group_size) {
        std::vector<uint8_t> q_data(expert_num * rows * cols);
        std::vector<ov::float16> scales;
        std::vector<uint8_t> zps;

        size_t num_groups_per_row = rows / group_size;
        size_t num_groups = cols * num_groups_per_row;
        float max_range = config.is_u4 ? 7.0f : 255.0f;
        float zp_max_range = config.is_u4 ? 127.0f : 255.0f;

        scales.resize(expert_num * num_groups);
        zps.resize(expert_num * num_groups);

        for (size_t e = 0; e < expert_num; ++e) {
            size_t offset = e * rows * cols;
            for (size_t c = 0; c < cols; ++c) {
                for (size_t g = 0; g < num_groups_per_row; ++g) {
                    float min_val = std::numeric_limits<float>::max();
                    float max_val = std::numeric_limits<float>::lowest();

                    size_t group_start = g * group_size;
                    for (size_t i = 0; i < group_size; ++i) {
                        float val = data[offset + (group_start + i) * cols + c];
                        if (val < min_val)
                            min_val = val;
                        if (val > max_val)
                            max_val = val;
                    }

                    float scale = (max_val - min_val) / max_range;
                    if (scale < 1e-5f)
                        scale = 1e-5f;
                    float zp = -min_val / scale;

                    uint8_t zp_val = static_cast<uint8_t>(std::round(std::max(0.0f, std::min(zp_max_range, zp))));

                    // Scales: [expert_num, num_groups_per_row, cols](Row-Major)
                    scales[e * num_groups + g * cols + c] = static_cast<ov::float16>(scale);
                    // ZPs: [expert_num, num_groups_per_row, cols](Row-Major)
                    zps[e * num_groups + g * cols + c] = zp_val;

                    for (size_t i = 0; i < group_size; ++i) {
                        float val = data[offset + (group_start + i) * cols + c];
                        float q_val = val / scale + zp_val;
                        uint8_t q = static_cast<uint8_t>(std::round(std::max(0.0f, std::min(max_range, q_val))));

                        // Store at [expert_num, cols, row] (Cols-Major)
                        size_t r = group_start + i;
                        q_data[offset + c * rows + r] = q;
                    }
                }
            }
        }
        return {q_data, scales, zps};
    }

    std::vector<ov::float16> run_reference(const std::vector<ov::float16>& hidden_states,
                                           const std::vector<ov::float16>& routing_weights,
                                           const std::vector<float>& w0_data,
                                           const std::vector<float>& w1_data,
                                           const std::vector<float>& w2_data) {
        size_t batch_size = config.batch_size;
        size_t seq_len = config.seq_len;
        size_t hidden_size = config.hidden_size;
        size_t inter_size = config.inter_size;
        size_t num_experts = config.num_experts;
        size_t top_k = config.top_k;

        std::vector<ov::float16> output(batch_size * seq_len * hidden_size, 0);

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                // Do softmax for routing weights
                std::vector<float> logits(num_experts);
                float max_val = std::numeric_limits<float>::lowest();
                for (size_t e = 0; e < num_experts; ++e) {
                    logits[e] = static_cast<float>(routing_weights[b * seq_len * num_experts + s * num_experts + e]);
                    if (logits[e] > max_val)
                        max_val = logits[e];
                }

                float sum_exp = 0.0f;
                for (size_t e = 0; e < num_experts; ++e) {
                    logits[e] = std::exp(logits[e] - max_val);
                    sum_exp += logits[e];
                }

                // Get routing weights and top-k
                std::vector<std::pair<float, size_t>> expert_weights;
                for (size_t e = 0; e < num_experts; ++e) {
                    expert_weights.push_back({logits[e] / sum_exp, e});
                }
                // Sort descending
                std::partial_sort(expert_weights.begin(),
                                  expert_weights.begin() + top_k,
                                  expert_weights.end(),
                                  [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
                                      return a.first > b.first;
                                  });

                // Normalize weights
                float sum_weights = 0.0f;
                for (size_t k = 0; k < top_k; ++k) {
                    sum_weights += expert_weights[k].first;
                }

                for (size_t k = 0; k < top_k; ++k) {
                    size_t expert_idx = expert_weights[k].second;
                    float weight = expert_weights[k].first / sum_weights;

                    std::vector<float> gate(inter_size);
                    std::vector<float> up(inter_size);

                    // Compute gate and up
                    for (size_t i = 0; i < inter_size; ++i) {
                        float gate_val = 0.0f;
                        float up_val = 0.0f;
                        for (size_t j = 0; j < hidden_size; ++j) {
                            float x_val = static_cast<float>(hidden_states[b * seq_len * hidden_size + s * hidden_size + j]);

                            // w0_data is Row-Major: [num_experts, hidden_size, inter_size]
                            size_t w0_cols = inter_size;
                            size_t w0_r = expert_idx * hidden_size + j;
                            float w0_val = w0_data[w0_r * w0_cols + i];
                            gate_val += x_val * w0_val;

                            // w1_data is Row-Major: [num_experts, hidden_size, inter_size]
                            size_t w1_cols = inter_size;
                            size_t w1_r = expert_idx * hidden_size + j;
                            float w1_val = w1_data[w1_r * w1_cols + i];
                            up_val += x_val * w1_val;
                        }
                        gate[i] = gate_val;
                        up[i] = up_val;
                    }

                    // SwiGLU
                    std::vector<float> act(inter_size);
                    for (size_t i = 0; i < inter_size; ++i) {
                        float silu = gate[i] / (1.0f + std::exp(-gate[i]));
                        act[i] = silu * up[i];
                    }

                    // Compute out = act @ w2
                    for (size_t j = 0; j < hidden_size; ++j) {
                        float out_val = 0.0f;
                        for (size_t i = 0; i < inter_size; ++i) {
                            // w2_data is Row-Major: [num_experts, inter_size, hidden_size]
                            size_t w2_cols = hidden_size;
                            size_t w2_r = expert_idx * inter_size + i;
                            float w2_val = w2_data[w2_r * w2_cols + j];

                            out_val += act[i] * w2_val;
                        }
                        output[b * seq_len * hidden_size + s * hidden_size + j] += static_cast<ov::float16>(out_val * weight);
                    }
                }
            }
        }
        return output;
    }
};

struct Moe3GemmTestParams {
    size_t seq_len;
    bool is_u4;
    size_t hidden_size;
    size_t inter_size;
    size_t num_experts;
    size_t top_k;
    size_t group_size;
};

class moe_3gemm_compressed_gpu_random : public ::testing::TestWithParam<Moe3GemmTestParams> {};

TEST_P(moe_3gemm_compressed_gpu_random, moe_accuracy_test_random) {
    auto param = GetParam();
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        return;
    }

    tests::random_generator rg(GET_SUITE_NAME);
    Moe3GemmConfig config;
    config.batch_size = 1;
    config.seq_len = param.seq_len;
    config.hidden_size = param.hidden_size;
    config.inter_size = param.inter_size;
    config.num_experts = param.num_experts;
    config.top_k = param.top_k;
    config.group_size = param.group_size;
    config.is_u4 = param.is_u4;

    Moe3GemmReference ref(config, rg);

    // Generate random data
    auto hidden_states = rg.generate_random_1d<ov::float16>(config.batch_size * config.seq_len * config.hidden_size, -1.0f, 1.0f, 1000);
    auto routing_weights = rg.generate_random_1d<ov::float16>(config.batch_size * config.seq_len * config.num_experts, 0.0f, 1.0f, 1000);

    auto w0_data = rg.generate_random_1d<float>(config.num_experts * config.hidden_size * config.inter_size, -1.0f, 0.0f, 1000);
    auto w1_data = rg.generate_random_1d<float>(config.num_experts * config.hidden_size * config.inter_size, -1.0f, 0.0f, 1000);
    auto w2_data = rg.generate_random_1d<float>(config.num_experts * config.inter_size * config.hidden_size, -1.0f, 0.0f, 1000);
    for (size_t i = 0; i < config.num_experts * config.hidden_size * config.inter_size; ++i) {
        w0_data[i] /= 7.0f;
        w1_data[i] /= 11.0f;
        w2_data[i] /= 7.0f;
    }
    auto [w0_q, w0_scale, w0_zp] = ref.quantize(w0_data, config.num_experts, config.hidden_size, config.inter_size, config.group_size);
    auto [w1_q, w1_scale, w1_zp] = ref.quantize(w1_data, config.num_experts, config.hidden_size, config.inter_size, config.group_size);
    auto [w2_q, w2_scale, w2_zp] = ref.quantize(w2_data, config.num_experts, config.inter_size, config.hidden_size, config.group_size);

    auto w0_q_packed = ref.pack(w0_q);
    auto w0_zp_packed = ref.pack(w0_zp);
    auto w1_q_packed = ref.pack(w1_q);
    auto w1_zp_packed = ref.pack(w1_zp);
    auto w2_q_packed = ref.pack(w2_q);
    auto w2_zp_packed = ref.pack(w2_zp);

    // Create tensors
    auto create_weight_tensor = [&](const std::vector<uint8_t>& values, int64_t b, int64_t f, int64_t y, int64_t x) {
        auto dt = config.is_u4 ? data_types::u4 : data_types::u8;
        auto mem = engine.allocate_memory({dt, format::bfyx, {b, f, y, x}});
        set_values(mem, values);
        get_test_stream().finish();
        return mem;
    };

    auto create_zp_tensor = [&](const std::vector<uint8_t>& values, int64_t b, int64_t f, int64_t y, int64_t x) {
        auto dt = config.is_u4 ? data_types::u4 : data_types::u8;
        auto mem = engine.allocate_memory({dt, format::bfyx, {b, f, y, x}});
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

    auto hidden_states_mem = create_f16_tensor(hidden_states, config.batch_size, config.seq_len, config.hidden_size, 1);
    auto routing_weights_mem = create_f16_tensor(routing_weights, config.batch_size, config.seq_len, config.num_experts, 1);

    size_t group_num = config.hidden_size / config.group_size;
    size_t group_num2 = config.inter_size / config.group_size;

    auto w0_weight_mem = create_weight_tensor(w0_q_packed, config.num_experts, config.inter_size, config.group_size, group_num);
    auto w0_scale_mem = create_f16_tensor(w0_scale, config.num_experts, group_num, 1, config.inter_size);
    auto w0_zp_mem = create_zp_tensor(w0_zp_packed, config.num_experts, group_num, 1, config.inter_size);

    auto w1_weight_mem = create_weight_tensor(w1_q_packed, config.num_experts, config.inter_size, config.group_size, group_num);
    auto w1_scale_mem = create_f16_tensor(w1_scale, config.num_experts, group_num, 1, config.inter_size);
    auto w1_zp_mem = create_zp_tensor(w1_zp_packed, config.num_experts, group_num, 1, config.inter_size);

    auto w2_weight_mem = create_weight_tensor(w2_q_packed, config.num_experts, config.hidden_size, config.group_size, group_num2);
    auto w2_scale_mem = create_f16_tensor(w2_scale, config.num_experts, group_num2, 1, config.hidden_size);
    auto w2_zp_mem = create_zp_tensor(w2_zp_packed, config.num_experts, group_num2, 1, config.hidden_size);

    // Build topology
    topology topology;
    topology.add(input_layout("hidden_states", hidden_states_mem->get_layout()));
    topology.add(input_layout("routing_weights", routing_weights_mem->get_layout()));
    topology.add(data("w0_weight", w0_weight_mem));
    topology.add(data("w0_scale", w0_scale_mem));
    topology.add(data("w0_zp", w0_zp_mem));
    topology.add(data("w1_weight", w1_weight_mem));
    topology.add(data("w1_scale", w1_scale_mem));
    topology.add(data("w1_zp", w1_zp_mem));
    topology.add(data("w2_weight", w2_weight_mem));
    topology.add(data("w2_scale", w2_scale_mem));
    topology.add(data("w2_zp", w2_zp_mem));

    cldnn::MOE3GemmFusedCompressed::Config moe_config;
    moe_config.hidden_size = config.hidden_size;
    moe_config.inter_size = config.inter_size;
    moe_config.num_expert = config.num_experts;
    moe_config.top_k = config.top_k;
    moe_config.group_size = config.group_size;
    moe_config.out_type = data_types::f16;

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
                                               moe_config);

    topology.add(moe_prim);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("hidden_states", hidden_states_mem);
    network.set_input_data("routing_weights", routing_weights_mem);

    auto outputs = network.execute();
    auto output_prim = outputs.begin()->second.get_memory();
    get_test_stream().flush();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_prim, get_test_stream());

    auto ref_output = ref.run_reference(hidden_states, routing_weights, w0_data, w1_data, w2_data);
    for (size_t i = 0; i < ref_output.size(); ++i) {
        ASSERT_NEAR(static_cast<float>(output_ptr[i]), static_cast<float>(ref_output[i]), 0.1f);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         moe_3gemm_compressed_gpu_random,
                         ::testing::Values(Moe3GemmTestParams{1, true, 128, 256, 4, 2, 128},
                                           Moe3GemmTestParams{16, true, 128, 256, 4, 2, 128},
                                           Moe3GemmTestParams{1, false, 128, 256, 4, 2, 128},
                                           Moe3GemmTestParams{16, false, 128, 256, 4, 2, 128}));

TEST(moe_3gemm_compressed_gpu, moe_accuracy_test_u4) {
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
    auto w1_zp = create_u4_tensor(w1_zp_data, num_experts, inter_size, group_num, 1);

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