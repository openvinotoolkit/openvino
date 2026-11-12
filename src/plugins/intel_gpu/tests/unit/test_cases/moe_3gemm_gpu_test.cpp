// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/moe_3gemm_fused_compressed.hpp>
#include <intel_gpu/primitives/moe_router_fused.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/layout.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/les_context.hpp>
#include <iostream>
#include <cstring>
#include <numeric>
#include <sstream>

#include "../../common/random_generator.hpp"
#include "../test_utils/test_utils.h"
#include "moe_3gemm_fused_inst.h"
#include "moe_3gemm_test_data.h"
#include "common_test_utils/common_utils.hpp"
#include "primitive_inst.h"
#include "ocl_v2/moe/residency/moe_lookahead_registry.hpp"
#include "ocl/kernels_cache.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"

using namespace cldnn;
using namespace ::tests;

class moe_3gemm_offload_fallback : public ::testing::TestWithParam<std::tuple<size_t, bool, bool, bool>> {
protected:
    std::filesystem::path weights_path;

    void TearDown() override {
        if (!weights_path.empty()) {
            std::error_code error;
            std::filesystem::remove(weights_path, error);
            EXPECT_FALSE(error) << error.message();
        }
    }
};

TEST_P(moe_3gemm_offload_fallback, one_slot_matches_reference) {
    const auto [initial_tokens, is_u4, shared, enable_les] = GetParam();
    size_t tokens = initial_tokens ? initial_tokens : 33;
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "No immad support";
    }
    auto& stream = get_test_stream();
    constexpr size_t hidden = 128;
    constexpr size_t experts = 4;
    constexpr size_t top_k = 2;
    const auto weight_type = is_u4 ? data_types::u4 : data_types::u8;
    topology topo;
    auto input = engine.allocate_memory(layout{ov::Shape{1, tokens, hidden}, data_types::f16, format::bfyx});
    auto routing = engine.allocate_memory(layout{ov::Shape{1, tokens, top_k}, data_types::f16, format::bfyx});
    auto indices = engine.allocate_memory(layout{ov::Shape{1, tokens, top_k}, data_types::i32, format::bfyx});
    const ov::Dimension sequence = initial_tokens ? ov::Dimension(tokens) : ov::Dimension::dynamic();
    topo.add(input_layout("hidden", layout{ov::PartialShape{1, sequence, hidden}, data_types::f16, format::bfyx}));
    topo.add(input_layout("routing", layout{ov::PartialShape{1, sequence, top_k}, data_types::f16, format::bfyx}));
    topo.add(input_layout("indices", layout{ov::PartialShape{1, sequence, top_k}, data_types::i32, format::bfyx}));
    std::vector<input_info> inputs{input_info("hidden"), input_info("routing"), input_info("indices")};

    weights_path = std::filesystem::temp_directory_path() / (ov::test::utils::generateTestFilePrefix() + ".bin");
    std::ofstream file(weights_path, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    std::vector<size_t> offsets(9);
    size_t file_offset = 0;
    for (size_t part = 0; part < (shared ? 2U : 1U); ++part) {
        const bool routed = part == 0;
        const size_t count = routed ? experts : 1;
        for (size_t projection = 0; projection < 3; ++projection) {
            for (size_t kind = 0; kind < 3; ++kind) {
                const auto type = kind == 1 ? data_types::f16 : weight_type;
                const ov::Shape shape = kind == 0 ? ov::Shape{count, hidden, 1, hidden}
                                                 : ov::Shape{count, hidden, 1};
                const layout full_layout{shape, type, format::bfyx};
                auto full = engine.allocate_memory(full_layout);
                if (kind == 1) {
                    set_values(full, std::vector<ov::float16>(count * hidden, ov::float16(0.01f)));
                } else {
                    std::vector<uint8_t> bytes(full_layout.bytes_count(), 0);
                    const size_t expert_bytes = bytes.size() / count;
                    if (kind == 0) {
                        for (size_t e = 0; e < count; ++e) {
                            const uint8_t value = static_cast<uint8_t>(routed ? e + projection + 1 : 2);
                            std::fill_n(bytes.begin() + e * expert_bytes,
                                        expert_bytes,
                                        is_u4 ? static_cast<uint8_t>(value | (value << 4)) : value);
                        }
                    }
                    set_values(full, bytes);
                }
                stream.finish();
                const std::string name = "weight_" + std::to_string(part) + "_" +
                                         std::to_string(projection) + "_" + std::to_string(kind);
                if (routed) {
                    offsets[kind * 3 + projection] = file_offset;
                    mem_lock<uint8_t, mem_lock_type::read> bytes(full, stream);
                    file.write(reinterpret_cast<const char*>(bytes.data()), full_layout.bytes_count());
                    ASSERT_TRUE(file.good());
                    file_offset += full_layout.bytes_count();
                    auto slot_shape = shape;
                    slot_shape[0] = 1;
                    auto slot = engine.allocate_memory(layout{slot_shape, type, format::bfyx},
                                                       engine.get_preferred_memory_allocation_type(),
                                                       false);
                    ASSERT_EQ(slot->size(), full_layout.bytes_count() / experts);
                    topo.add(data(name, engine.reinterpret_buffer(*slot, full_layout), true));
                } else {
                    topo.add(data(name, full));
                }
                inputs.emplace_back(name);
            }
        }
    }
    file.close();
    ASSERT_FALSE(file.fail());
    if (shared) {
        auto scalar = engine.allocate_memory(layout{ov::PartialShape{1, hidden}, data_types::f16, format::bfyx});
        set_values(scalar, std::vector<ov::float16>(hidden, ov::float16(0)));
        topo.add(data("shared_scalar", scalar));
        inputs.emplace_back("shared_scalar");
    }

    MOECompressed::Config config{};
    config.hidden_size = hidden;
    config.inter_size = hidden;
    config.num_expert = experts;
    config.top_k = top_k;
    config.group_size = hidden;
    config.num_shared_expert = shared ? 1 : 0;
    config.has_zp = true;
    config.out_type = data_types::f16;
    topo.add(moe_3gemm_fused_compressed("moe", inputs, config, offsets, weights_path, 1));
    auto net_config = get_test_default_config(engine);
    net_config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    net_config.set_property({ov::intel_gpu::moe_lookahead_prefetch(enable_les),
                             ov::intel_gpu::les_queue_capacity(2),
                             ov::intel_gpu::offload_ratio(50)});
    network net(engine, topo, net_config);
    auto lookahead = ov::intel_gpu::ocl::moe::LookaheadRegistry::acquire(net);
    EXPECT_EQ(lookahead != nullptr, enable_les);
    if (enable_les && tokens == 1 && is_u4 && !shared) {
        cldnn::network independent(net.get_program());
        auto other = ov::intel_gpu::ocl::moe::LookaheadRegistry::acquire(independent);
        ASSERT_NE(other, nullptr);
        EXPECT_NE(lookahead, other);
        auto validate = [&](const ExecutionConfig& candidate) {
            cldnn::network invalid_network(engine, topo, candidate);
            (void)ov::intel_gpu::ocl::moe::LookaheadRegistry::acquire(invalid_network);
        };
        auto invalid = net_config.clone();
        invalid.set_property(ov::num_streams(2));
        EXPECT_THROW(validate(invalid), ov::Exception);
        invalid = net_config.clone();
        invalid.set_property(ov::intel_gpu::offload_ratio(100));
        EXPECT_THROW(validate(invalid), ov::Exception);
        invalid = net_config.clone();
        invalid.set_property(ov::intel_gpu::record_replay(true));
        EXPECT_THROW(validate(invalid), ov::Exception);
    }
    net.set_input_data("hidden", input);
    net.set_input_data("routing", routing);
    net.set_input_data("indices", indices);

    const float scale = static_cast<float>(ov::float16(0.01f));
    const std::vector<size_t> transition_tokens{33, 1, 16, 1, 33, 1};
    const size_t iterations = initial_tokens ? 3 : transition_tokens.size();
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        SCOPED_TRACE(iteration);
        if (!initial_tokens) {
            tokens = transition_tokens[iteration];
            input = engine.allocate_memory(layout{ov::Shape{1, tokens, hidden}, data_types::f16, format::bfyx});
            routing = engine.allocate_memory(layout{ov::Shape{1, tokens, top_k}, data_types::f16, format::bfyx});
            indices = engine.allocate_memory(layout{ov::Shape{1, tokens, top_k}, data_types::i32, format::bfyx});
            net.set_input_data("hidden", input);
            net.set_input_data("routing", routing);
            net.set_input_data("indices", indices);
        }
        const int32_t first_expert = iteration == 1 ? 2 : 0;
        const ov::float16 x(0.1f * static_cast<float>(iteration + 1));
        set_values(input, std::vector<ov::float16>(tokens * hidden, x));
        std::vector<int32_t> ids(tokens * top_k);
        std::vector<ov::float16> routes(tokens * top_k);
        for (size_t t = 0; t < tokens; ++t) {
            ids[t * top_k] = first_expert;
            ids[t * top_k + 1] = first_expert + 1;
            routes[t * top_k] = ov::float16(0.25f);
            routes[t * top_k + 1] = ov::float16(0.75f);
        }
        set_values(indices, ids);
        set_values(routing, routes);
        stream.finish();
        const auto outputs = net.execute();
        EXPECT_EQ(ov::intel_gpu::ocl::moe::LookaheadRegistry::acquire(net), lookahead);
        if (enable_les) {
            auto& instance = static_cast<typed_primitive_inst<moe_3gemm_fused_compressed>&>(*net.get_primitive("moe"));
            const auto& weights = instance._weights;
            const std::vector<memory::ptr> bound{weights.gate_w, weights.gate_s, weights.gate_z,
                weights.up_w, weights.up_s, weights.up_z, weights.down_w, weights.down_s, weights.down_z};
            constexpr size_t first_weight_input = 3;
            for (size_t index = 0; index < bound.size(); ++index) {
                EXPECT_EQ(bound[index], instance.input_memory_ptr(first_weight_input + index));
            }
        }
        auto output = outputs.at("moe").get_memory();
        mem_lock<ov::float16, mem_lock_type::read> result(output, stream);
        auto expert_output = [&](float gate_weight, float up_weight, float down_weight) {
            const float gate = hidden * static_cast<float>(x) * gate_weight * scale;
            const float up = hidden * static_cast<float>(x) * up_weight * scale;
            return hidden * (gate / (1.0f + std::exp(-gate))) * up * down_weight * scale;
        };
        const float e = static_cast<float>(first_expert);
        float expected = 0.25f * expert_output(e + 1, e + 2, e + 3) +
                         0.75f * expert_output(e + 2, e + 3, e + 4);
        if (shared) {
            expected += 0.5f * expert_output(2, 2, 2);
        }
        ASSERT_EQ(output->get_layout().count(), tokens * hidden);
        for (size_t i = 0; i < tokens * hidden; ++i) {
            ASSERT_NEAR(static_cast<float>(result[i]), expected, std::abs(expected) * 0.01f + 0.002f) << i;
        }
        mem_lock<ov::float16, mem_lock_type::read> routing_result(routing, stream);
        for (size_t i = 0; i < routes.size(); ++i) {
            EXPECT_EQ(routing_result[i], routes[i]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         moe_3gemm_offload_fallback,
                         ::testing::Combine(::testing::Values(size_t{1}, size_t{2}, size_t{16}, size_t{33}, size_t{0}),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool()));

struct Moe3GemmConfig {
    size_t batch_size;
    size_t seq_len;
    size_t hidden_size;
    size_t inter_size;
    size_t num_experts;
    size_t top_k;
    size_t group_size;
    bool is_u4;
    bool is_signed = false;
    bool has_shared_expert = false;
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

    // Quantized-then-dequantized weights differ from the original ones by ~7% RMS for u4 (~0.3% for u8),
    // enough to exceed the tolerance on its own.
    std::vector<float> dequantize(const std::vector<uint8_t>& q_data,
                                  const std::vector<ov::float16>& scales,
                                  const std::vector<uint8_t>& zps,
                                  size_t expert_num,
                                  size_t rows,
                                  size_t cols,
                                  size_t group_size) {
        std::vector<float> out(expert_num * rows * cols);
        size_t num_groups_per_row = rows / group_size;
        size_t num_groups = cols * num_groups_per_row;
        for (size_t e = 0; e < expert_num; ++e) {
            size_t offset = e * rows * cols;
            for (size_t c = 0; c < cols; ++c) {
                for (size_t r = 0; r < rows; ++r) {
                    size_t g = r / group_size;
                    float s = static_cast<float>(scales[e * num_groups + g * cols + c]);
                    float z = static_cast<float>(zps[e * num_groups + g * cols + c]);
                    out[offset + r * cols + c] = (static_cast<float>(q_data[offset + c * rows + r]) - z) * s;
                }
            }
        }
        return out;
    }

    std::tuple<std::vector<uint8_t>, std::vector<ov::float16>> quantize_symmetric(const std::vector<float>& data,
                                                                                  size_t expert_num,
                                                                                  size_t rows,
                                                                                  size_t cols,
                                                                                  size_t group_size) {
        std::vector<uint8_t> q_data(expert_num * rows * cols);
        std::vector<ov::float16> scales;

        size_t num_groups_per_row = rows / group_size;
        size_t num_groups = cols * num_groups_per_row;
        int max_range = config.is_u4 ? 7 : 127;
        int min_range = config.is_u4 ? -8 : -128;

        scales.resize(expert_num * num_groups);

        for (size_t e = 0; e < expert_num; ++e) {
            size_t offset = e * rows * cols;
            for (size_t c = 0; c < cols; ++c) {
                for (size_t g = 0; g < num_groups_per_row; ++g) {
                    float max_abs = 0.0f;
                    size_t group_start = g * group_size;

                    for (size_t i = 0; i < group_size; ++i) {
                        float val = std::abs(data[offset + (group_start + i) * cols + c]);
                        if (val > max_abs)
                            max_abs = val;
                    }

                    float scale = max_abs / static_cast<float>(max_range);
                    if (scale < 1e-5f)
                        scale = 1e-5f;

                    // Scales: [expert_num, num_groups_per_row, cols](Row-Major)
                    scales[e * num_groups + g * cols + c] = static_cast<ov::float16>(scale);

                    for (size_t i = 0; i < group_size; ++i) {
                        float val = data[offset + (group_start + i) * cols + c];
                        float q_val = val / scale;
                        int q = static_cast<int>(std::round(std::max(static_cast<float>(min_range), std::min(static_cast<float>(max_range), q_val))));

                        // Store at [expert_num, cols, row] (Cols-Major) as two's complement
                        size_t r = group_start + i;
                        q_data[offset + c * rows + r] = static_cast<uint8_t>(static_cast<int8_t>(q));
                    }
                }
            }
        }
        return {q_data, scales};
    }

    std::vector<ov::float16> run_reference_softmax(const std::vector<ov::float16>& hidden_states,
                                                   const std::vector<ov::float16>& routing_weights,
                                                   const std::vector<float>& w0_data,
                                                   const std::vector<float>& w1_data,
                                                   const std::vector<float>& w2_data,
                                                   const std::vector<float>& s_gate_data = {},
                                                   const std::vector<float>& s_up_data = {},
                                                   const std::vector<float>& s_down_data = {},
                                                   const std::vector<float>& s_gate_scalar_data = {}) {
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

                // Normalize top-k weights
                float sum_weights = 0.0f;
                for (size_t k = 0; k < top_k; ++k)
                    sum_weights += expert_weights[k].first;

                std::vector<std::pair<float, size_t>> top_k_normalized(top_k);
                for (size_t k = 0; k < top_k; ++k)
                    top_k_normalized[k] = {expert_weights[k].first / sum_weights, expert_weights[k].second};
                apply_top_k_experts(b, s, hidden_states, w0_data, w1_data, w2_data, top_k_normalized, output);

                // Shared Expert
                if (config.has_shared_expert) {
                    std::vector<float> shared_gate(inter_size);
                    std::vector<float> shared_up(inter_size);
                    float shared_scalar_gate = 0.0f;

                    // Compute scalar gate
                    for (size_t j = 0; j < hidden_size; ++j) {
                        float x = static_cast<float>(hidden_states[b * seq_len * hidden_size + s * hidden_size + j]);
                        shared_scalar_gate += x * s_gate_scalar_data[j];
                    }
                    shared_scalar_gate = 1.0f / (1.0f + std::exp(-shared_scalar_gate));

                    // Compute gate and up
                    for (size_t i = 0; i < inter_size; ++i) {
                        float gate_val = 0.0f;
                        float up_val = 0.0f;
                        for (size_t j = 0; j < hidden_size; ++j) {
                            float x_val = static_cast<float>(hidden_states[b * seq_len * hidden_size + s * hidden_size + j]);
                            gate_val += x_val * s_gate_data[j * inter_size + i];
                            up_val += x_val * s_up_data[j * inter_size + i];
                        }
                        shared_gate[i] = gate_val;
                        shared_up[i] = up_val;
                    }

                    std::vector<float> shared_act(inter_size);
                    for (size_t i = 0; i < inter_size; ++i) {
                        float silu = shared_gate[i] / (1.0f + std::exp(-shared_gate[i]));
                        shared_act[i] = silu * shared_up[i];
                    }

                    // Compute out = act @ s_down * scalar_gate
                    for (size_t j = 0; j < hidden_size; ++j) {
                        float out_val = 0.0f;
                        for (size_t i = 0; i < inter_size; ++i) {
                            out_val += shared_act[i] * s_down_data[i * hidden_size + j];
                        }
                        output[b * seq_len * hidden_size + s * hidden_size + j] += static_cast<ov::float16>(out_val * shared_scalar_gate);
                    }
                }
            }
        }
        return output;
    }

    std::vector<ov::float16> run_reference_sigmoid(const std::vector<ov::float16>& hidden_states,
                                                   const std::vector<ov::float16>& routing_logits,
                                                   const std::vector<ov::float16>& routing_bias,
                                                   ov::float16 epsilon,
                                                   const std::vector<float>& w0_data,
                                                   const std::vector<float>& w1_data,
                                                   const std::vector<float>& w2_data) {
        size_t batch_size = config.batch_size;
        size_t seq_len = config.seq_len;
        size_t num_experts = config.num_experts;
        size_t top_k = config.top_k;

        std::vector<ov::float16> output(batch_size * seq_len * config.hidden_size, 0);

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                // Apply sigmoid to logits
                std::vector<float> sigmoid_scores(num_experts);
                for (size_t e = 0; e < num_experts; ++e) {
                    float logit = static_cast<float>(routing_logits[b * seq_len * num_experts + s * num_experts + e]);
                    sigmoid_scores[e] = 1.0f / (1.0f + std::exp(-logit));
                }

                // Add bias and select top-k by (sigmoid + bias)
                std::vector<std::pair<float, size_t>> expert_weights;
                for (size_t e = 0; e < num_experts; ++e) {
                    float score = sigmoid_scores[e] + static_cast<float>(routing_bias[e]);
                    expert_weights.push_back({score, e});
                }
                std::partial_sort(expert_weights.begin(),
                                  expert_weights.begin() + top_k,
                                  expert_weights.end(),
                                  [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
                                      return a.first > b.first;
                                  });

                // Gather sigmoid scores at top-k indices and normalize
                float sum_weights = 0.0f;
                for (size_t k = 0; k < top_k; ++k)
                    sum_weights += sigmoid_scores[expert_weights[k].second];
                sum_weights += static_cast<float>(epsilon);

                std::vector<std::pair<float, size_t>> top_k_normalized(top_k);
                for (size_t k = 0; k < top_k; ++k)
                    top_k_normalized[k] = {sigmoid_scores[expert_weights[k].second] / sum_weights, expert_weights[k].second};
                apply_top_k_experts(b, s, hidden_states, w0_data, w1_data, w2_data, top_k_normalized, output);
            }
        }
        return output;
    }

private:
    void apply_top_k_experts(size_t b,
                             size_t s,
                             const std::vector<ov::float16>& hidden_states,
                             const std::vector<float>& w0_data,
                             const std::vector<float>& w1_data,
                             const std::vector<float>& w2_data,
                             const std::vector<std::pair<float, size_t>>& top_k_normalized,
                             std::vector<ov::float16>& output) {
        const size_t hidden_size = config.hidden_size;
        const size_t inter_size = config.inter_size;

        for (const auto& [weight, expert_idx] : top_k_normalized) {
            std::vector<float> gate(inter_size);
            std::vector<float> up(inter_size);

            // Compute gate and up
            for (size_t i = 0; i < inter_size; ++i) {
                float gate_val = 0.0f;
                float up_val = 0.0f;
                for (size_t j = 0; j < hidden_size; ++j) {
                    float x_val = static_cast<float>(hidden_states[b * config.seq_len * hidden_size + s * hidden_size + j]);

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
                output[b * config.seq_len * hidden_size + s * hidden_size + j] += static_cast<ov::float16>(out_val * weight);
            }
        }
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
    bool is_signed = false;
    bool enable_les = false;
    bool profiling = false;
    bool exhaustive_masks = false;
    size_t lru_expert_num = 0;
    bool proxy_router = false;
    bool proxy_fc_bias = false;
    bool proxy_perfect_recall = false;
    bool proxy_prefetch = false;
    bool proxy_compressed = false;
    bool proxy_runtime_kernels = false;
};

class scoped_moe_env {
public:
    scoped_moe_env(const char* name, const char* value) : _name(name) {
        if (const auto* previous = std::getenv(name)) {
            _had_previous = true;
            _previous = previous;
        }
#ifdef _WIN32
        _putenv_s(name, value);
#else
        ::setenv(name, value, 1);
#endif
    }
    ~scoped_moe_env() {
#ifdef _WIN32
        _putenv_s(_name.c_str(), _had_previous ? _previous.c_str() : "");
#else
        if (_had_previous) {
            ::setenv(_name.c_str(), _previous.c_str(), 1);
        } else {
            ::unsetenv(_name.c_str());
        }
#endif
    }
    scoped_moe_env(const scoped_moe_env&) = delete;
    scoped_moe_env& operator=(const scoped_moe_env&) = delete;

private:
    std::string _name;
    std::string _previous;
    bool _had_previous = false;
};

class moe_3gemm_compressed_gpu_random : public ::testing::TestWithParam<std::tuple<cldnn::MoERouterFused::RoutingType, Moe3GemmTestParams>> {
protected:
    std::filesystem::path offload_weights_path;
    void TearDown() override {
        if (!offload_weights_path.empty()) {
            std::error_code error;
            std::filesystem::remove(offload_weights_path, error);
            EXPECT_FALSE(error) << error.message();
        }
    }
};

TEST_P(moe_3gemm_compressed_gpu_random, moe_accuracy_test_random) {
    const auto& [routing_type, param] = GetParam();
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "No immad support";
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

    auto w0_qdq = ref.dequantize(w0_q, w0_scale, w0_zp, config.num_experts, config.hidden_size, config.inter_size, config.group_size);
    auto w1_qdq = ref.dequantize(w1_q, w1_scale, w1_zp, config.num_experts, config.hidden_size, config.inter_size, config.group_size);
    auto w2_qdq = ref.dequantize(w2_q, w2_scale, w2_zp, config.num_experts, config.inter_size, config.hidden_size, config.group_size);

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


    auto create_f16_tensor = [&](const std::vector<ov::float16>& values, int64_t b, int64_t f, int64_t y, int64_t x) {
        auto mem = engine.allocate_memory({data_types::f16, format::bfyx, {b, f, y, x}});
        set_values(mem, values);
        get_test_stream().finish();
        return mem;
    };

    auto create_f16_tensor_3d = [&](const std::vector<ov::float16>& values, int64_t d0, int64_t d1, int64_t d2) {
        auto mem = engine.allocate_memory(layout{ov::PartialShape{d0, d1, d2}, data_types::f16, format::bfyx});
        set_values(mem, values);
        get_test_stream().finish();
        return mem;
    };

    auto hidden_states_mem = create_f16_tensor_3d(hidden_states, config.batch_size, config.seq_len, config.hidden_size);
    auto routing_weights_mem = create_f16_tensor_3d(routing_weights, config.batch_size, config.seq_len, config.num_experts);
    auto routing_bias_data = rg.generate_random_1d<ov::float16>(config.num_experts, -0.5f, 0.5f, 1000);
    auto routing_bias_mem = create_f16_tensor(routing_bias_data, 1, 1, 1, config.num_experts);
    ov::float16 routing_eps_val = ov::float16(1e-6f);

    size_t group_num = config.hidden_size / config.group_size;
    size_t group_num2 = config.inter_size / config.group_size;

    // Logical [E, ofm, G]; physical [E, G, ofm] via byfx for G>1, plain bfyx for per-channel.
    auto make_scale = [&](const std::vector<ov::float16>& v, int64_t E, int64_t ofm, int64_t G) {
        auto fmt = G > 1 ? format::byfx : format::bfyx;
        auto shape = G > 1 ? ov::PartialShape{E, ofm, G, 1} : ov::PartialShape{E, ofm, 1};
        auto mem = engine.allocate_memory(layout{shape, data_types::f16, fmt});
        set_values(mem, v);
        get_test_stream().finish();
        return mem;
    };
    auto make_zp = [&](const std::vector<uint8_t>& v, int64_t E, int64_t ofm, int64_t G) {
        auto dt = config.is_u4 ? data_types::u4 : data_types::u8;
        auto fmt = G > 1 ? format::byfx : format::bfyx;
        auto shape = G > 1 ? ov::PartialShape{E, ofm, G, 1} : ov::PartialShape{E, ofm, 1};
        auto mem = engine.allocate_memory(layout{shape, dt, fmt});
        set_values(mem, v);
        get_test_stream().finish();
        return mem;
    };

    auto w0_weight_mem = create_weight_tensor(w0_q_packed, config.num_experts, config.inter_size, config.group_size, group_num);
    auto w0_scale_mem = make_scale(w0_scale, config.num_experts, config.inter_size, group_num);
    auto w0_zp_mem = make_zp(w0_zp_packed, config.num_experts, config.inter_size, group_num);

    auto w1_weight_mem = create_weight_tensor(w1_q_packed, config.num_experts, config.inter_size, config.group_size, group_num);
    auto w1_scale_mem = make_scale(w1_scale, config.num_experts, config.inter_size, group_num);
    auto w1_zp_mem = make_zp(w1_zp_packed, config.num_experts, config.inter_size, group_num);

    auto w2_weight_mem = create_weight_tensor(w2_q_packed, config.num_experts, config.hidden_size, config.group_size, group_num2);
    auto w2_scale_mem = make_scale(w2_scale, config.num_experts, config.hidden_size, group_num2);
    auto w2_zp_mem = make_zp(w2_zp_packed, config.num_experts, config.hidden_size, group_num2);

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

    // Create MoERouterFused primitive
    MoERouterFused::Config router_config;
    router_config.num_expert = config.num_experts;
    router_config.top_k = config.top_k;
    std::vector<input_info> router_inputs{input_info("routing_weights")};
    if (routing_type == cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS) {
        router_config.routing_type = MoERouterFused::RoutingType::SIGMOID_BIAS;
        topology.add(data("routing_bias", routing_bias_mem));
        auto routing_eps_mem = engine.allocate_memory({data_types::f16, format::bfyx, {1, 1, 1, 1}});
        set_values(routing_eps_mem, {routing_eps_val});
        get_test_stream().finish();
        topology.add(data("routing_eps", routing_eps_mem));
        router_inputs.push_back(input_info("routing_bias"));
        router_inputs.push_back(input_info("routing_eps"));
    }
    topology.add(moe_router_fused("router", router_inputs, router_config));

    // Create MOE3GemmFusedCompressed primitive
    cldnn::MOECompressed::Config moe_config;
    moe_config.hidden_size = config.hidden_size;
    moe_config.inter_size = config.inter_size;
    moe_config.num_expert = config.num_experts;
    moe_config.top_k = config.top_k;
    moe_config.group_size = config.group_size;
    moe_config.out_type = data_types::f16;
    moe_config.has_zp = true;

    std::vector<input_info> moe_inputs{input_info("hidden_states"),
                                       input_info("router", 0),  // topk_weights
                                       input_info("router", 1),  // topk_indices
                                       input_info("w0_weight"),
                                       input_info("w0_scale"),
                                       input_info("w0_zp"),
                                       input_info("w1_weight"),
                                       input_info("w1_scale"),
                                       input_info("w1_zp"),
                                       input_info("w2_weight"),
                                       input_info("w2_scale"),
                                       input_info("w2_zp")};

    topology.add(moe_3gemm_fused_compressed("moe_3gemm_fused_compressed", moe_inputs, moe_config));

    if (param.proxy_router) {
        auto router_weight_memory = engine.allocate_memory(
            layout{ov::PartialShape{static_cast<int64_t>(config.num_experts), static_cast<int64_t>(config.hidden_size)},
                   param.proxy_compressed ? (param.is_u4 ? data_types::u4 : data_types::u8) : data_types::f16,
                   format::bfyx});
        auto router_weights = param.proxy_perfect_recall
                      ? std::vector<ov::float16>(config.num_experts * config.hidden_size, ov::float16(0.0f))
                      : rg.generate_random_1d<ov::float16>(config.num_experts * config.hidden_size, -0.25f, 0.25f, 1000);
        if (param.proxy_compressed) {
            auto quantized = rg.generate_random_1d<uint8_t>(router_weights.size(), 0, 15);
            if (param.proxy_perfect_recall) {
                std::fill(quantized.begin(), quantized.end(), 8);
            }
            set_values(router_weight_memory, ref.pack(quantized));
            const size_t groups = config.hidden_size / 64;
            auto scales = engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(config.num_experts),
                                                                        static_cast<int64_t>(groups)},
                                                       data_types::f16, format::fbyx});
            set_values(scales, rg.generate_random_1d<ov::float16>(config.num_experts * groups, 0.01f, 0.1f, 1000));
            auto zeros = engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(config.num_experts),
                                                                       static_cast<int64_t>(groups)},
                                                      data_types::u8, format::fbyx});
            set_values(zeros, std::vector<uint8_t>(config.num_experts * groups, 8));
            topology.add(data("proxy_router_scales", scales), data("proxy_router_zeros", zeros));
        } else {
            set_values(router_weight_memory, router_weights);
        }
        get_test_stream().finish();
        topology.add(data("proxy_router_weights", router_weight_memory));
        std::string proxy_projection_bias_id;
        if (param.proxy_fc_bias) {
            proxy_projection_bias_id = "proxy_router_projection_bias";
            auto projection_bias = engine.allocate_memory(
                layout{ov::PartialShape{static_cast<int64_t>(config.num_experts)}, data_types::f16, format::bfyx});
            set_values(projection_bias, rg.generate_random_1d<ov::float16>(config.num_experts, -0.1f, 0.1f, 1000));
            get_test_stream().finish();
            topology.add(data(proxy_projection_bias_id, projection_bias));
        }
        if (param.proxy_compressed) {
            topology.add(fully_connected("proxy_router_projection",
                                         input_info("moe_3gemm_fused_compressed"),
                                         "proxy_router_weights", proxy_projection_bias_id,
                                         "proxy_router_scales", "proxy_router_zeros", data_types::f16, 3));
        } else {
            topology.add(fully_connected("proxy_router_projection",
                                     input_info("moe_3gemm_fused_compressed"),
                                     "proxy_router_weights",
                                     proxy_projection_bias_id,
                                     3));
        }
        MoERouterFused::Config proxy_router_config;
        proxy_router_config.num_expert = config.num_experts;
        proxy_router_config.top_k = config.top_k;
        proxy_router_config.routing_type = routing_type;
        std::vector<input_info> proxy_router_inputs{input_info("proxy_router_projection")};
        if (routing_type == MoERouterFused::RoutingType::SIGMOID_BIAS) {
            auto proxy_routing_bias = engine.allocate_memory(
                layout{ov::PartialShape{static_cast<int64_t>(config.num_experts)}, data_types::f16, format::bfyx});
            set_values(proxy_routing_bias, routing_bias_data);
            get_test_stream().finish();
            topology.add(data("proxy_routing_bias", proxy_routing_bias));
            auto proxy_routing_eps = engine.allocate_memory(
                layout{ov::PartialShape{1}, data_types::f16, format::bfyx});
            set_values(proxy_routing_eps, {routing_eps_val});
            get_test_stream().finish();
            topology.add(data("proxy_routing_eps", proxy_routing_eps));
            proxy_router_inputs.emplace_back("proxy_routing_bias");
            proxy_router_inputs.emplace_back("proxy_routing_eps");
        }
        topology.add(moe_router_fused("proxy_router", proxy_router_inputs, proxy_router_config));
        auto proxy_target_inputs = moe_inputs;
        proxy_target_inputs[0] = input_info("moe_3gemm_fused_compressed");
        proxy_target_inputs[1] = input_info("proxy_router", 0);
        proxy_target_inputs[2] = input_info("proxy_router", 1);
        ASSERT_EQ(group_num, 1U);
        ASSERT_EQ(group_num2, 1U);
        offload_weights_path = std::filesystem::temp_directory_path() /
            ("moe_proxy_les_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
        std::ofstream file(offload_weights_path, std::ios::binary);
        ASSERT_TRUE(file.is_open());
        std::vector<size_t> offsets;
        for (const auto& memory : std::vector<memory::ptr>{w0_weight_mem, w1_weight_mem, w2_weight_mem,
                                                         w0_scale_mem, w1_scale_mem, w2_scale_mem,
                                                         w0_zp_mem, w1_zp_mem, w2_zp_mem}) {
            offsets.push_back(static_cast<size_t>(file.tellp()));
            std::vector<uint8_t> bytes(memory->size());
            memory->copy_to(get_test_stream(), bytes.data(), 0, 0, bytes.size(), true);
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
        file.close();
        ASSERT_TRUE(file.good());
        const std::vector<memory::ptr> target_weights{w0_weight_mem, w0_scale_mem, w0_zp_mem,
                                                     w1_weight_mem, w1_scale_mem, w1_zp_mem,
                                                     w2_weight_mem, w2_scale_mem, w2_zp_mem};
        for (size_t index = 0; index < target_weights.size(); ++index) {
            auto full_layout = target_weights[index]->get_layout();
            auto partial_shape = full_layout.get_partial_shape();
            partial_shape[0] = param.lru_expert_num;
            auto partial_layout = full_layout;
            partial_layout.set_partial_shape(partial_shape);
            auto backing = engine.allocate_memory(partial_layout, engine.get_preferred_memory_allocation_type(), false);
            const auto id = "proxy_partial_" + std::to_string(index);
            topology.add(data(id, engine.reinterpret_buffer(*backing, full_layout), true));
            proxy_target_inputs[3 + index] = input_info(id);
        }
        auto target = moe_3gemm_fused_compressed("proxy_target", proxy_target_inputs, moe_config);
        target._otd.lru_expert_num = param.lru_expert_num;
        target._otd.weight_bin_offsets = offsets;
        target._otd.weights_path = offload_weights_path;
        topology.add(target);
    }

    auto net_config = get_test_default_config(engine);
    net_config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    if (param.enable_les) {
        net_config.set_property({ov::intel_gpu::moe_lookahead_prefetch(true),
                                 ov::enable_profiling(param.profiling),
                                 ov::intel_gpu::les_queue_capacity(2),
                                 ov::intel_gpu::offload_ratio(param.proxy_router ? 50 : 0),
                                 ov::num_streams(1)});
    }
    std::unique_ptr<scoped_moe_env> prediction_env;
    std::unique_ptr<scoped_moe_env> prediction_compare_env;
    std::unique_ptr<scoped_moe_env> prediction_perf_env;
    std::unique_ptr<scoped_moe_env> prediction_prefetch_env;
    if (param.proxy_router) {
        net_config.set_property(ov::weights_path(offload_weights_path.string()));
        prediction_env = std::make_unique<scoped_moe_env>("OV_GPU_LES_PREDICT", "1");
        prediction_compare_env = std::make_unique<scoped_moe_env>("OV_GPU_LES_PREDICT_COMPARE", "1");
        prediction_perf_env = std::make_unique<scoped_moe_env>("MOE_OTD_PERF_LOG", "1");
        prediction_prefetch_env = std::make_unique<scoped_moe_env>("OV_GPU_LES_PREFETCH", param.proxy_prefetch ? "1" : "0");
    }
    cldnn::topology baseline_topology;
    if (param.proxy_router) {
        for (const auto& entry : topology.get_primitives()) {
            baseline_topology.add_primitive(entry.second->clone());
        }
    }
    network network(engine, topology, net_config);
    auto lookahead = ov::intel_gpu::ocl::moe::LookaheadRegistry::acquire(network);
    if (!param.proxy_router) {
        EXPECT_EQ(lookahead, nullptr);
    }
    network.set_input_data("hidden_states", hidden_states_mem);
    network.set_input_data("routing_weights", routing_weights_mem);

    if (param.proxy_router) {
        ASSERT_NE(lookahead, nullptr);
        auto links = lookahead->links();
        ASSERT_EQ(links.size(), 2U);
        EXPECT_EQ(links.back().status, program::les_router_status::supported);
        EXPECT_EQ(links.back().chain.source_moe, "moe_3gemm_fused_compressed");
        EXPECT_EQ(links.back().chain.target_moe, "proxy_target");
        EXPECT_EQ(links.back().router, "proxy_router");
        const auto* next_layer = lookahead->get_next_layer(network.get_primitive("moe_3gemm_fused_compressed").get());
        ASSERT_NE(next_layer, nullptr);
        EXPECT_TRUE(next_layer->is_supported);
        EXPECT_EQ(next_layer->moe_inst, network.get_primitive("proxy_target").get());
        EXPECT_EQ(next_layer->router_fc_inst, network.get_primitive(links.back().projection).get());
        EXPECT_EQ(next_layer->weight_mem, network.get_primitive(links.back().weights)->output_memory_ptr());
        EXPECT_EQ(lookahead->get_next_layer(network.get_primitive("proxy_target").get()), nullptr);
        EXPECT_EQ(lookahead->get_next_layer(nullptr), nullptr);
        EXPECT_TRUE(network.get_config().get_moe_lookahead_prefetch());
        if (param.proxy_runtime_kernels) {
            auto router = network.get_primitive("proxy_router");
            auto binaries = router->get_impl()->get_kernels();
            cldnn::kernels_cache::compiled_kernels compiled;
            auto& kernels = compiled[*router->get_impl_params()];
            for (size_t index = 0; index < binaries.size(); ++index) {
                kernels.emplace_back(binaries[index], index);
            }
            router->get_impl()->set_kernels(std::move(compiled));
        }
        auto predicted_outputs = network.execute();
        const auto state = lookahead->prediction("proxy_target");
        ASSERT_NE(state.hidden, nullptr);
        ASSERT_NE(state.hidden_ready, nullptr);
        ASSERT_NE(state.prediction_ready, nullptr);
        ASSERT_NE(state.predicted_indices, nullptr);
        EXPECT_TRUE(engine.is_the_same_buffer(*state.hidden,
            *network.get_primitive("moe_3gemm_fused_compressed")->input_memory_ptr(0)));
        state.prediction_ready->wait();
        auto actual_indices = network.get_primitive("proxy_router")->output_memory_ptr(1);
        EXPECT_FALSE(engine.is_the_same_buffer(*state.predicted_indices, *actual_indices));
        mem_lock<uint32_t, mem_lock_type::read> predicted_ids(state.predicted_indices, network.get_stream());
        mem_lock<uint32_t, mem_lock_type::read> actual_ids(actual_indices, network.get_stream());
        size_t prediction_matches = 0;
        for (size_t index = 0; index < config.top_k; ++index) {
            EXPECT_LT(predicted_ids[index], config.num_experts) << index;
            EXPECT_LT(actual_ids[index], config.num_experts) << index;
            prediction_matches += predicted_ids[index] == actual_ids[index];
        }
        EXPECT_LE(prediction_matches, config.top_k);
        if (param.proxy_perfect_recall) {
            EXPECT_EQ(prediction_matches, config.top_k);
        }
        std::vector<ov::float16> prior_source_output(config.hidden_size);
        network.get_primitive("moe_3gemm_fused_compressed")->output_memory_ptr()->copy_to(
            network.get_stream(), prior_source_output.data(), 0, 0, prior_source_output.size() * sizeof(ov::float16), true);

        auto baseline_config = net_config.clone();
        baseline_config.set_property(ov::intel_gpu::moe_lookahead_prefetch(false));
        cldnn::network baseline(engine, baseline_topology, baseline_config);
        baseline.set_input_data("hidden_states", hidden_states_mem);
        baseline.set_input_data("routing_weights", routing_weights_mem);
        auto baseline_outputs = baseline.execute();
        auto baseline_result = baseline_outputs.begin()->second.get_memory();
        auto predicted_result = predicted_outputs.begin()->second.get_memory();
        mem_lock<ov::float16, mem_lock_type::read> baseline_values(baseline_result, baseline.get_stream());
        mem_lock<ov::float16, mem_lock_type::read> predicted_values(predicted_result, network.get_stream());
        for (size_t index = 0; index < config.hidden_size; ++index) {
            EXPECT_EQ(std::memcmp(&baseline_values[index], &predicted_values[index], sizeof(ov::float16)), 0) << index;
        }

        if (param.proxy_prefetch) {
            std::stringstream serialized;
            BinaryOutputBuffer output_buffer(serialized);
            output_buffer.set_stream(network.get_stream_ptr().get());
            network.get_program()->save(output_buffer);
            serialized.seekg(0);
            BinaryInputBuffer input_buffer(serialized, engine);
            auto imported_program = std::make_shared<program>(engine, net_config);
            imported_program->load(input_buffer);
            cldnn::network imported(imported_program);
            auto imported_lookahead = ov::intel_gpu::ocl::moe::LookaheadRegistry::acquire(imported);
            ASSERT_NE(imported_lookahead, nullptr);
            auto imported_links = imported_lookahead->links();
            ASSERT_EQ(imported_links.size(), 2U);
            EXPECT_EQ(imported_links.back().status, program::les_router_status::supported);
            imported.set_input_data("hidden_states", hidden_states_mem);
            imported.set_input_data("routing_weights", routing_weights_mem);
            auto imported_outputs = imported.execute();
            const auto imported_state = imported_lookahead->prediction("proxy_target");
            ASSERT_NE(imported_state.prediction_ready, nullptr);
            imported_state.prediction_ready->wait();
            auto imported_result = imported_outputs.begin()->second.get_memory();
            mem_lock<ov::float16, mem_lock_type::read> imported_values(imported_result, imported.get_stream());
            for (size_t index = 0; index < config.hidden_size; ++index) {
                EXPECT_EQ(std::memcmp(&baseline_values[index], &imported_values[index], sizeof(ov::float16)), 0)
                    << "Imported proxy output mismatch " << index;
            }
        }

        auto prior_snapshot = state.hidden;
        auto prior_prediction = state.prediction_ready;
        std::vector<uint32_t> prior_proxy_ids(config.top_k);
        std::vector<uint32_t> prior_router_ids(config.top_k);
        state.prediction_ready->wait();
        state.predicted_indices->copy_to(network.get_stream(), prior_proxy_ids.data(), 0, 0,
                                         prior_proxy_ids.size() * sizeof(uint32_t), true);
        network.get_primitive("proxy_router")->output_memory_ptr(1)->copy_to(
            network.get_stream(), prior_router_ids.data(), 0, 0, prior_router_ids.size() * sizeof(uint32_t), true);
        auto next_outputs = network.execute();
        auto next_state = lookahead->prediction("proxy_target");
        ASSERT_NE(next_state.hidden, nullptr);
        ASSERT_NE(next_state.prediction_ready, nullptr);
        EXPECT_TRUE(engine.is_the_same_buffer(*next_state.hidden, *prior_snapshot));
        EXPECT_NE(next_state.prediction_ready, prior_prediction);
        next_state.prediction_ready->wait();
        std::vector<uint32_t> next_proxy_ids(config.top_k);
        std::vector<uint32_t> next_router_ids(config.top_k);
        next_state.predicted_indices->copy_to(network.get_stream(), next_proxy_ids.data(), 0, 0,
                                              next_proxy_ids.size() * sizeof(uint32_t), true);
        network.get_primitive("proxy_router")->output_memory_ptr(1)->copy_to(
            network.get_stream(), next_router_ids.data(), 0, 0, next_router_ids.size() * sizeof(uint32_t), true);
        EXPECT_EQ(next_proxy_ids, prior_proxy_ids);
        EXPECT_EQ(next_router_ids, prior_router_ids);
        auto next_result = next_outputs.begin()->second.get_memory();
        mem_lock<ov::float16, mem_lock_type::read> next_values(next_result, network.get_stream());
        for (size_t index = 0; index < config.hidden_size; ++index) {
            EXPECT_EQ(std::memcmp(&baseline_values[index], &next_values[index], sizeof(ov::float16)), 0)
                << "Repeated proxy output mismatch " << index;
        }
        return;
    }

    auto les_links = network.get_program()->get_les_moe_links();
    if (param.enable_les) {
        ASSERT_EQ(les_links.size(), 1U);
        EXPECT_EQ(les_links.front().target_moe, "moe_3gemm_fused_compressed");
        EXPECT_EQ(les_links.front().status, cldnn::program::les_link_status::no_predecessor);
        EXPECT_FALSE(les_links.front().target_hidden.empty());
        EXPECT_TRUE(les_links.front().source_moe.empty());
    } else {
        EXPECT_TRUE(les_links.empty());
    }

    auto outputs = network.execute();
    EXPECT_EQ(lookahead, nullptr);
    auto output_prim = outputs.begin()->second.get_memory();
    get_test_stream().flush();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_prim, get_test_stream());

    auto ref_output = routing_type == cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS
                          ? ref.run_reference_sigmoid(hidden_states, routing_weights, routing_bias_data, routing_eps_val, w0_qdq, w1_qdq, w2_qdq)
                          : ref.run_reference_softmax(hidden_states, routing_weights, w0_qdq, w1_qdq, w2_qdq);
    // SigmoidBias routing performs all routing math (sigmoid, bias, normalization) in f16 on the kernel side,
    // while the reference uses f32, leading to slightly larger numerical divergence than Softmax.
    const float base_tolerance = routing_type == cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS ? 0.3f : 0.15f;
    const float tolerance = base_tolerance * (config.hidden_size / 128);
    for (size_t i = 0; i < ref_output.size(); ++i) {
        ASSERT_NEAR(static_cast<float>(output_ptr[i]), static_cast<float>(ref_output[i]), tolerance);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         moe_3gemm_compressed_gpu_random,
                         ::testing::Combine(::testing::Values(cldnn::MoERouterFused::RoutingType::SOFTMAX,
                                                              cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS),
                                            ::testing::Values(Moe3GemmTestParams{1, true, 128, 256, 4, 2, 128},
                                                              Moe3GemmTestParams{16, true, 128, 256, 4, 2, 128},
                                                              Moe3GemmTestParams{1, false, 128, 256, 4, 2, 128},
                                                              Moe3GemmTestParams{16, false, 128, 256, 4, 2, 128},
                                                              Moe3GemmTestParams{1, true, 256, 512, 4, 2, 256},
                                                              Moe3GemmTestParams{1, false, 256, 512, 4, 2, 256},
                                                              Moe3GemmTestParams{1, true, 512, 512, 4, 2, 512},
                                                              Moe3GemmTestParams{1, false, 512, 512, 4, 2, 512},
                                                              Moe3GemmTestParams{33, true, 128, 256, 4, 2, 128},
                                                              Moe3GemmTestParams{33, false, 128, 256, 4, 2, 128},
                                                              Moe3GemmTestParams{48, true, 128, 256, 4, 2, 128},
                                                              Moe3GemmTestParams{48, false, 128, 256, 4, 2, 128})));

INSTANTIATE_TEST_SUITE_P(smoke_les_p4_proxy,
    moe_3gemm_compressed_gpu_random,
    ::testing::Combine(
        ::testing::Values(cldnn::MoERouterFused::RoutingType::SOFTMAX,
                          cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS),
        ::testing::ValuesIn([] {
            std::vector<Moe3GemmTestParams> parameters;
            for (size_t variant = 0; variant < 4; ++variant) {
                Moe3GemmTestParams parameter{1, true, 128, 128, 16, 8, 128};
                parameter.enable_les = true;
                parameter.lru_expert_num = 8;
                parameter.proxy_router = true;
                parameter.proxy_fc_bias = variant == 1;
                parameter.proxy_perfect_recall = variant >= 2;
                parameter.proxy_prefetch = variant == 3;
                parameters.push_back(parameter);
            }
            return parameters;
        }())));

INSTANTIATE_TEST_SUITE_P(smoke_les_resident,
    moe_3gemm_compressed_gpu_random,
    ::testing::Combine(
        ::testing::Values(cldnn::MoERouterFused::RoutingType::SOFTMAX, cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS),
        ::testing::Values(
            Moe3GemmTestParams{1, true, 256, 256, 4, 2, 128, false, true},
            Moe3GemmTestParams{4, true, 256, 256, 4, 2, 128, false, true})));

INSTANTIATE_TEST_SUITE_P(smoke_les_p4_compressed_proxy,
    moe_3gemm_compressed_gpu_random,
    ::testing::Combine(
        ::testing::Values(cldnn::MoERouterFused::RoutingType::SOFTMAX, cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS),
        ::testing::ValuesIn([] {
            std::vector<Moe3GemmTestParams> parameters;
            for (bool u4 : {false, true}) {
                for (bool perfect : {false, true}) {
                    Moe3GemmTestParams parameter{1, u4, 128, 128, 16, 8, 128};
                    parameter.enable_les = true;
                    parameter.lru_expert_num = 8;
                    parameter.proxy_router = true;
                    parameter.proxy_compressed = true;
                    parameter.proxy_runtime_kernels = true;
                    parameter.proxy_perfect_recall = perfect;
                    parameters.push_back(parameter);
                }
            }
            return parameters;
        }())));

INSTANTIATE_TEST_SUITE_P(smoke_les_p4_runtime_proxy,
    moe_3gemm_compressed_gpu_random,
    ::testing::Combine(
        ::testing::Values(cldnn::MoERouterFused::RoutingType::SOFTMAX, cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS),
        ::testing::ValuesIn([] {
            std::vector<Moe3GemmTestParams> parameters;
            for (bool bias : {false, true}) {
                Moe3GemmTestParams parameter{1, true, 128, 128, 16, 8, 128};
                parameter.enable_les = true;
                parameter.lru_expert_num = 8;
                parameter.proxy_router = true;
                parameter.proxy_fc_bias = bias;
                parameter.proxy_runtime_kernels = true;
                parameters.push_back(parameter);
            }
            Moe3GemmTestParams small{1, true, 128, 128, 4, 2, 128};
            small.enable_les = true;
            small.profiling = true;
            small.lru_expert_num = 2;
            small.proxy_router = true;
            small.proxy_runtime_kernels = true;
            parameters.push_back(small);
            return parameters;
        }())));
// Sub-128 weight quantization group size (e.g. trinity-mini afmoe uses group_size=64
// with SIGMOID_BIAS routing). Limited to SIGMOID_BIAS to mirror real model usage and
// to avoid known accuracy noise of the SOFTMAX path on some platforms.
INSTANTIATE_TEST_SUITE_P(smoke_sub128_group_size,
                         moe_3gemm_compressed_gpu_random,
                         ::testing::Combine(::testing::Values(cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS),
                                            ::testing::Values(Moe3GemmTestParams{1, true, 128, 256, 4, 2, 64},
                                                              Moe3GemmTestParams{16, true, 128, 256, 4, 2, 64},
                                                              Moe3GemmTestParams{1, false, 128, 256, 4, 2, 64},
                                                              Moe3GemmTestParams{1, true, 256, 512, 4, 2, 64},
                                                              Moe3GemmTestParams{1, false, 256, 512, 4, 2, 64})));

// Batched GEMV: MTP/speculative decoding scenarios where token_num is 2-16.
// These small batch sizes exercise the batched GEMV path (exec_batched_gemv)
// which avoids gather/scatter/CPU-sync overhead of the prefill GEMM path.
INSTANTIATE_TEST_SUITE_P(smoke_batched_gemv_mtp,
                         moe_3gemm_compressed_gpu_random,
                         ::testing::Combine(::testing::Values(cldnn::MoERouterFused::RoutingType::SOFTMAX,
                                                              cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS),
                                            ::testing::Values(Moe3GemmTestParams{2, true, 128, 256, 4, 2, 128},
                                                              Moe3GemmTestParams{4, true, 128, 256, 4, 2, 128},
                                                              Moe3GemmTestParams{8, true, 128, 256, 4, 2, 128},
                                                              Moe3GemmTestParams{2, false, 128, 256, 4, 2, 128},
                                                              Moe3GemmTestParams{4, false, 128, 256, 4, 2, 128},
                                                              Moe3GemmTestParams{8, false, 128, 256, 4, 2, 128},
                                                              Moe3GemmTestParams{2, true, 256, 512, 4, 2, 256},
                                                              Moe3GemmTestParams{4, true, 256, 512, 4, 2, 256},
                                                              Moe3GemmTestParams{2, false, 256, 512, 4, 2, 256},
                                                              Moe3GemmTestParams{4, false, 256, 512, 4, 2, 256},
                                                              // Sub-128 group_size batched GEMV coverage.
                                                              Moe3GemmTestParams{2, true, 128, 256, 4, 2, 64},
                                                              Moe3GemmTestParams{2, false, 128, 256, 4, 2, 64})));

// Batched GEMV with a hidden_size that is not a multiple of 1024 (Gemma-4-26B-A4B: 2816).
// The mlp_reduce local size must divide hidden_size; some drivers reject non-uniform work-groups.
INSTANTIATE_TEST_SUITE_P(smoke_batched_gemv_hidden_not_multiple_of_1024,
                         moe_3gemm_compressed_gpu_random,
                         ::testing::Combine(::testing::Values(cldnn::MoERouterFused::RoutingType::SOFTMAX,
                                                              cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS),
                                            ::testing::Values(Moe3GemmTestParams{1, true, 2816, 512, 4, 2, 128},
                                                              Moe3GemmTestParams{2, true, 2816, 512, 4, 2, 128})));

class moe_3gemm_compressed_gpu_u4 : public ::testing::TestWithParam<cldnn::MoERouterFused::RoutingType> {};

class moe_3gemm_compressed_gpu_shared_random : public ::testing::TestWithParam<Moe3GemmTestParams> {
protected:
    std::filesystem::path offload_weights_path;
    void TearDown() override {
        if (!offload_weights_path.empty()) {
            std::error_code error;
            std::filesystem::remove(offload_weights_path, error);
            EXPECT_FALSE(error) << error.message();
        }
    }
};

TEST_P(moe_3gemm_compressed_gpu_shared_random, moe_accuracy_test_shared_expert_random) {
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
    config.has_shared_expert = true;

    Moe3GemmReference ref(config, rg);

    // Generate random data
    auto hidden_states = rg.generate_random_1d<ov::float16>(config.batch_size * config.seq_len * config.hidden_size, -1.0f, 1.0f, 1000);
    auto routing_weights = rg.generate_random_1d<ov::float16>(config.batch_size * config.seq_len * config.num_experts, 0.0f, 1.0f, 1000);

    auto w0_data = rg.generate_random_1d<float>(config.num_experts * config.hidden_size * config.inter_size, -1.0f, 0.0f, 1000);
    auto w1_data = rg.generate_random_1d<float>(config.num_experts * config.hidden_size * config.inter_size, -1.0f, 0.0f, 1000);
    auto w2_data = rg.generate_random_1d<float>(config.num_experts * config.inter_size * config.hidden_size, -1.0f, 0.0f, 1000);

    // Shared Expert Data
    auto s_gate_data = rg.generate_random_1d<float>(config.hidden_size * config.inter_size, -1.0f, 0.0f, 1000);
    auto s_up_data = rg.generate_random_1d<float>(config.hidden_size * config.inter_size, -1.0f, 0.0f, 1000);    // [hidden, inter]
    auto s_down_data = rg.generate_random_1d<float>(config.inter_size * config.hidden_size, -1.0f, 0.0f, 1000);  // [inter, hidden]
    auto s_gate_scalar_data = rg.generate_random_1d<float>(config.hidden_size, -1.0f, 0.0f, 1000);

    for (size_t i = 0; i < config.num_experts * config.hidden_size * config.inter_size; ++i) {
        w0_data[i] /= 7.0f;
        w1_data[i] /= 11.0f;
        w2_data[i] /= 7.0f;
        if (i < config.hidden_size * config.inter_size) {
            s_gate_data[i] /= 7.0f;
            s_up_data[i] /= 11.0f;
            s_down_data[i] /= 7.0f;
        }
    }

    // Quantize Experts
    auto [w0_q, w0_scale, w0_zp] = ref.quantize(w0_data, config.num_experts, config.hidden_size, config.inter_size, config.group_size);
    auto [w1_q, w1_scale, w1_zp] = ref.quantize(w1_data, config.num_experts, config.hidden_size, config.inter_size, config.group_size);
    auto [w2_q, w2_scale, w2_zp] = ref.quantize(w2_data, config.num_experts, config.inter_size, config.hidden_size, config.group_size);

    // Quantize Shared Experts (Treat as 1 expert)
    auto [s_gate_q, s_gate_scale, s_gate_zp] = ref.quantize(s_gate_data, 1, config.hidden_size, config.inter_size, config.group_size);
    auto [s_up_q, s_up_scale, s_up_zp] = ref.quantize(s_up_data, 1, config.hidden_size, config.inter_size, config.group_size);
    auto [s_down_q, s_down_scale, s_down_zp] = ref.quantize(s_down_data, 1, config.inter_size, config.hidden_size, config.group_size);

    auto w0_q_packed = ref.pack(w0_q);
    auto w0_zp_packed = ref.pack(w0_zp);
    auto w1_q_packed = ref.pack(w1_q);
    auto w1_zp_packed = ref.pack(w1_zp);
    auto w2_q_packed = ref.pack(w2_q);
    auto w2_zp_packed = ref.pack(w2_zp);

    auto s_gate_q_packed = ref.pack(s_gate_q);
    auto s_gate_zp_packed = ref.pack(s_gate_zp);
    auto s_up_q_packed = ref.pack(s_up_q);
    auto s_up_zp_packed = ref.pack(s_up_zp);
    auto s_down_q_packed = ref.pack(s_down_q);
    auto s_down_zp_packed = ref.pack(s_down_zp);

    // Create tensors
    auto create_weight_tensor = [&](const std::vector<uint8_t>& values, int64_t b, int64_t f, int64_t y, int64_t x) {
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

    // For Scalar Gate Weights - assuming 1D tensor [hidden_size]
    auto create_scalar_gate_tensor = [&](const std::vector<float>& values) {
        std::vector<ov::float16> fp16_values;
        for (float v : values)
            fp16_values.push_back(static_cast<ov::float16>(v));
        // Shape: [hidden_size]
        auto mem = engine.allocate_memory({data_types::f16, format::bfyx, {1, 1, 1, static_cast<int64_t>(values.size())}});
        set_values(mem, fp16_values);
        get_test_stream().finish();
        return mem;
    };

    auto hidden_states_mem = create_f16_tensor(hidden_states, config.batch_size, config.seq_len, config.hidden_size, 1);
    auto routing_weights_mem = create_f16_tensor(routing_weights, config.batch_size, config.seq_len, config.num_experts, 1);
    routing_weights_mem = engine.reinterpret_buffer(*routing_weights_mem,
        layout{ov::PartialShape{static_cast<int64_t>(config.batch_size), static_cast<int64_t>(config.seq_len),
                                 static_cast<int64_t>(config.num_experts)}, data_types::f16, format::bfyx});

    size_t group_num = config.hidden_size / config.group_size;
    size_t group_num2 = config.inter_size / config.group_size;

    // Expert Weights
    // Logical [E, ofm, G]; physical [E, G, ofm] via byfx for G>1, plain bfyx for per-channel.
    auto make_scale = [&](const std::vector<ov::float16>& v, int64_t E, int64_t ofm, int64_t G) {
        auto fmt = G > 1 ? format::byfx : format::bfyx;
        auto shape = G > 1 ? ov::PartialShape{E, ofm, G, 1} : ov::PartialShape{E, ofm, 1};
        auto mem = engine.allocate_memory(layout{shape, data_types::f16, fmt});
        set_values(mem, v);
        get_test_stream().finish();
        return mem;
    };
    auto make_zp = [&](const std::vector<uint8_t>& v, int64_t E, int64_t ofm, int64_t G) {
        auto dt = config.is_u4 ? data_types::u4 : data_types::u8;
        auto fmt = G > 1 ? format::byfx : format::bfyx;
        auto shape = G > 1 ? ov::PartialShape{E, ofm, G, 1} : ov::PartialShape{E, ofm, 1};
        auto mem = engine.allocate_memory(layout{shape, dt, fmt});
        set_values(mem, v);
        get_test_stream().finish();
        return mem;
    };

    auto w0_weight_mem = create_weight_tensor(w0_q_packed, config.num_experts, config.inter_size, config.group_size, group_num);
    auto w0_scale_mem = make_scale(w0_scale, config.num_experts, config.inter_size, group_num);
    auto w0_zp_mem = make_zp(w0_zp_packed, config.num_experts, config.inter_size, group_num);

    auto w1_weight_mem = create_weight_tensor(w1_q_packed, config.num_experts, config.inter_size, config.group_size, group_num);
    auto w1_scale_mem = make_scale(w1_scale, config.num_experts, config.inter_size, group_num);
    auto w1_zp_mem = make_zp(w1_zp_packed, config.num_experts, config.inter_size, group_num);

    auto w2_weight_mem = create_weight_tensor(w2_q_packed, config.num_experts, config.hidden_size, config.group_size, group_num2);
    auto w2_scale_mem = make_scale(w2_scale, config.num_experts, config.hidden_size, group_num2);
    auto w2_zp_mem = make_zp(w2_zp_packed, config.num_experts, config.hidden_size, group_num2);

    // Shared Expert Weights (num_experts = 1)
    auto s_gate_weight_mem = create_weight_tensor(s_gate_q_packed, 1, config.inter_size, config.group_size, group_num);
    auto s_gate_scale_mem = make_scale(s_gate_scale, 1, config.inter_size, group_num);
    auto s_gate_zp_mem = make_zp(s_gate_zp_packed, 1, config.inter_size, group_num);

    auto s_up_weight_mem = create_weight_tensor(s_up_q_packed, 1, config.inter_size, config.group_size, group_num);
    auto s_up_scale_mem = make_scale(s_up_scale, 1, config.inter_size, group_num);
    auto s_up_zp_mem = make_zp(s_up_zp_packed, 1, config.inter_size, group_num);

    auto s_down_weight_mem = create_weight_tensor(s_down_q_packed, 1, config.hidden_size, config.group_size, group_num2);
    auto s_down_scale_mem = make_scale(s_down_scale, 1, config.hidden_size, group_num2);
    auto s_down_zp_mem = make_zp(s_down_zp_packed, 1, config.hidden_size, group_num2);

    auto s_gate_scalar_mem = create_scalar_gate_tensor(s_gate_scalar_data);

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

    // Add shared inputs
    topology.add(data("s_gate_weight", s_gate_weight_mem));
    topology.add(data("s_gate_scale", s_gate_scale_mem));
    topology.add(data("s_gate_zp", s_gate_zp_mem));
    topology.add(data("s_up_weight", s_up_weight_mem));
    topology.add(data("s_up_scale", s_up_scale_mem));
    topology.add(data("s_up_zp", s_up_zp_mem));
    topology.add(data("s_down_weight", s_down_weight_mem));
    topology.add(data("s_down_scale", s_down_scale_mem));
    topology.add(data("s_down_zp", s_down_zp_mem));
    topology.add(data("s_gate_scalar", s_gate_scalar_mem));

    // Create MoERouterFused primitive (softmax for shared expert tests)
    MoERouterFused::Config router_config;
    router_config.num_expert = config.num_experts;
    router_config.top_k = config.top_k;
    router_config.routing_type = MoERouterFused::RoutingType::SOFTMAX;
    topology.add(moe_router_fused("router", {input_info("routing_weights")}, router_config));

    cldnn::MOECompressed::Config moe_config;
    moe_config.hidden_size = config.hidden_size;
    moe_config.inter_size = config.inter_size;
    moe_config.num_expert = config.num_experts;
    moe_config.top_k = config.top_k;
    moe_config.group_size = config.group_size;
    moe_config.out_type = data_types::f16;
    moe_config.num_shared_expert = 1;
    moe_config.has_zp = true;

    // Create Primitive with new input layout
    topology.add(moe_3gemm_fused_compressed("moe_3gemm_fused_compressed",
                                            {input_info("hidden_states"),
                                             input_info("router", 0),  // topk_weights
                                             input_info("router", 1),  // topk_indices
                                             input_info("w0_weight"),
                                             input_info("w0_scale"),
                                             input_info("w0_zp"),
                                             input_info("w1_weight"),
                                             input_info("w1_scale"),
                                             input_info("w1_zp"),
                                             input_info("w2_weight"),
                                             input_info("w2_scale"),
                                             input_info("w2_zp"),
                                             // Shared Expert Inputs (indices 12-21)
                                             input_info("s_gate_weight"),
                                             input_info("s_gate_scale"),
                                             input_info("s_gate_zp"),
                                             input_info("s_up_weight"),
                                             input_info("s_up_scale"),
                                             input_info("s_up_zp"),
                                             input_info("s_down_weight"),
                                             input_info("s_down_scale"),
                                             input_info("s_down_zp"),
                                             input_info("s_gate_scalar")},
                                            moe_config));

    auto net_config = get_test_default_config(engine);
    net_config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cldnn::topology offload_topology;
    if (param.enable_les) {
        for (const auto& entry : topology.get_primitives()) {
            offload_topology.add_primitive(entry.second->clone());
        }
    }
    network network(engine, topology, net_config);
    network.set_input_data("hidden_states", hidden_states_mem);
    network.set_input_data("routing_weights", routing_weights_mem);

    auto outputs = network.execute();
    auto output_prim = outputs.begin()->second.get_memory();
    get_test_stream().flush();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_prim, get_test_stream());

    auto ref_output =
        ref.run_reference_softmax(hidden_states, routing_weights, w0_data, w1_data, w2_data, s_gate_data, s_up_data, s_down_data, s_gate_scalar_data);

    // Check accuracy
    const float tolerance = 0.5f * (config.hidden_size / 128);
    for (size_t i = 0; i < ref_output.size(); ++i) {
        EXPECT_NEAR(static_cast<float>(output_ptr[i]), static_cast<float>(ref_output[i]), tolerance);
    }
    if (param.enable_les) {
        offload_weights_path = std::filesystem::temp_directory_path() /
            ("moe_shared_les_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
        std::ofstream file(offload_weights_path, std::ios::binary);
        ASSERT_TRUE(file.is_open());
        std::vector<size_t> offsets;
        for (const auto& memory : std::vector<memory::ptr>{w0_weight_mem, w1_weight_mem, w2_weight_mem,
                                                         w0_scale_mem, w1_scale_mem, w2_scale_mem,
                                                         w0_zp_mem, w1_zp_mem, w2_zp_mem}) {
            offsets.push_back(static_cast<size_t>(file.tellp()));
            std::vector<uint8_t> bytes(memory->size());
            memory->copy_to(get_test_stream(), bytes.data(), 0, 0, bytes.size(), true);
            if (memory->get_layout().format == format::byfx) {
                const auto shape = memory->get_layout().get_shape();
                const size_t channels = shape[1];
                const size_t groups = shape[2];
                const size_t bits = ov::element::Type(memory->get_layout().data_type).bitwidth();
                const auto grouped = bytes;
                for (size_t expert = 0; expert < config.num_experts; ++expert) {
                    for (size_t channel = 0; channel < channels; ++channel) {
                        for (size_t group = 0; group < groups; ++group) {
                            const size_t source = (expert * groups + group) * channels + channel;
                            const size_t destination = (expert * channels + channel) * groups + group;
                            if (bits == 4) {
                                const uint8_t value = (grouped[source / 2] >> ((source % 2) * 4)) & 0x0f;
                                const size_t shift = (destination % 2) * 4;
                                bytes[destination / 2] = static_cast<uint8_t>(
                                    (bytes[destination / 2] & ~(0x0f << shift)) | (value << shift));
                            } else {
                                std::memcpy(bytes.data() + destination * (bits / 8),
                                            grouped.data() + source * (bits / 8), bits / 8);
                            }
                        }
                    }
                }
            }
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
        file.close();
        ASSERT_TRUE(file.good());
        auto offloaded = *std::static_pointer_cast<moe_3gemm_fused_compressed>(offload_topology.at("moe_3gemm_fused_compressed"));
        offloaded._otd.lru_expert_num = param.lru_expert_num;
        offloaded._otd.weight_bin_offsets = offsets;
        offloaded._otd.weights_path = offload_weights_path;
        cldnn::topology split_topology;
        const std::vector<memory::ptr> routed_weights{w0_weight_mem, w0_scale_mem, w0_zp_mem,
                                                     w1_weight_mem, w1_scale_mem, w1_zp_mem,
                                                     w2_weight_mem, w2_scale_mem, w2_zp_mem};
        std::set<primitive_id> partial_ids;
        for (size_t index = 0; index < routed_weights.size(); ++index) {
            const auto id = offloaded.input[3 + index].pid;
            auto full_layout = routed_weights[index]->get_layout();
            auto partial_layout = full_layout;
            auto partial_shape = full_layout.get_partial_shape();
            partial_shape[0] = param.lru_expert_num;
            partial_layout.set_partial_shape(partial_shape);
            auto backing = engine.allocate_memory(partial_layout, engine.get_preferred_memory_allocation_type(), false);
            split_topology.add(data(id, engine.reinterpret_buffer(*backing, full_layout), true));
            partial_ids.insert(id);
        }
        for (const auto& entry : offload_topology.get_primitives()) {
            if (entry.first != offloaded.id && !partial_ids.count(entry.first)) {
                split_topology.add_primitive(entry.second);
            }
        }
        split_topology.add(offloaded);
        auto split_config = net_config.clone();
        split_config.set_property({ov::intel_gpu::moe_lookahead_prefetch(true), ov::intel_gpu::offload_ratio(50),
                                   ov::enable_profiling(param.profiling), ov::num_streams(1)});
        if (param.hidden_size == 2048) {
            split_config.set_property(ov::intel_gpu::queue_type(QueueTypes::out_of_order),
                                      ov::intel_gpu::use_onednn(false));
        }
        std::vector<ov::float16> expected_values(output_ptr.data(), output_ptr.data() + ref_output.size());
        if (param.seq_len > 1) {
            cldnn::topology control_topology;
            for (const auto& entry : split_topology.get_primitives()) {
                control_topology.add_primitive(entry.second->clone());
            }
            auto control_config = split_config.clone();
            control_config.set_property(ov::intel_gpu::moe_lookahead_prefetch(false));
            cldnn::network control(engine, control_topology, control_config);
            control.set_input_data("hidden_states", hidden_states_mem);
            control.set_input_data("routing_weights", routing_weights_mem);
            auto result = control.execute().begin()->second.get_memory();
            result->copy_to(control.get_stream(), expected_values.data(), 0, 0,
                            expected_values.size() * sizeof(ov::float16), true);
            for (size_t index = 0; index < ref_output.size(); ++index) {
                EXPECT_NEAR(static_cast<float>(expected_values[index]), static_cast<float>(ref_output[index]), tolerance);
            }
        }
        cldnn::network split(engine, split_topology, split_config);
        split.set_input_data("hidden_states", hidden_states_mem);
        split.set_input_data("routing_weights", routing_weights_mem);
        for (size_t iteration = 0; iteration < 3; ++iteration) {
            auto result = split.execute().begin()->second.get_memory();
            cldnn::mem_lock<ov::float16, mem_lock_type::read> values(result, split.get_stream());
            for (size_t index = 0; index < ref_output.size(); ++index) {
                ASSERT_EQ(std::memcmp(&values[index], &expected_values[index], sizeof(ov::float16)), 0)
                    << "Shared split mismatch " << index << ", iteration " << iteration;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_les_p3_shared,
    moe_3gemm_compressed_gpu_shared_random,
    ::testing::ValuesIn([] {
        std::vector<Moe3GemmTestParams> parameters;
        for (bool u4 : {false, true}) {
            for (size_t tokens : {1U, 4U}) {
                for (size_t capacity : {1U, 4U}) {
                    for (bool profiling : {false, true}) {
                        parameters.push_back({tokens, u4, 128, 128, 4, 2, 128, false, true, profiling, false, capacity});
                    }
                }
            }
        }
        for (bool profiling : {false, true}) {
            parameters.push_back({1, true, 2048, 512, 16, 8, 128, false, true, profiling, false, 8});
        }
        return parameters;
    }()));

INSTANTIATE_TEST_SUITE_P(smoke,
                         moe_3gemm_compressed_gpu_shared_random,
                         ::testing::Values(Moe3GemmTestParams{1, true, 128, 256, 4, 2, 128},
                                           Moe3GemmTestParams{16, true, 128, 256, 4, 2, 128},
                                           Moe3GemmTestParams{1, false, 128, 256, 4, 2, 128},
                                           Moe3GemmTestParams{16, false, 128, 256, 4, 2, 128},
                                           Moe3GemmTestParams{1, true, 256, 512, 4, 2, 256},
                                           Moe3GemmTestParams{1, false, 256, 512, 4, 2, 256},
                                           Moe3GemmTestParams{1, true, 512, 512, 4, 2, 512},
                                           Moe3GemmTestParams{1, false, 512, 512, 4, 2, 512},
                                           // Sub-128 group_size (trinity-mini afmoe shape).
                                           Moe3GemmTestParams{1, true, 128, 256, 4, 2, 64},
                                           Moe3GemmTestParams{1, false, 128, 256, 4, 2, 64},
                                           Moe3GemmTestParams{1, true, 256, 512, 4, 2, 64}));

// Batched GEMV with shared expert: MTP/speculative decoding scenarios with shared expert enabled.
// Exercises the batched GEMV path with EXPERTS_PER_TOKEN = MAX_TOPK + 1.
INSTANTIATE_TEST_SUITE_P(smoke_batched_gemv_mtp_shared,
                         moe_3gemm_compressed_gpu_shared_random,
                         ::testing::Values(Moe3GemmTestParams{2, true, 128, 256, 4, 2, 128},
                                           Moe3GemmTestParams{4, true, 128, 256, 4, 2, 128},
                                           Moe3GemmTestParams{8, true, 128, 256, 4, 2, 128},
                                           Moe3GemmTestParams{2, false, 128, 256, 4, 2, 128},
                                           Moe3GemmTestParams{4, false, 128, 256, 4, 2, 128},
                                           Moe3GemmTestParams{8, false, 128, 256, 4, 2, 128},
                                           Moe3GemmTestParams{2, true, 256, 512, 4, 2, 256},
                                           Moe3GemmTestParams{4, true, 256, 512, 4, 2, 256},
                                           Moe3GemmTestParams{2, false, 256, 512, 4, 2, 256},
                                           Moe3GemmTestParams{4, false, 256, 512, 4, 2, 256},
                                           // Sub-128 group_size batched GEMV coverage.
                                           Moe3GemmTestParams{2, true, 128, 256, 4, 2, 64},
                                           Moe3GemmTestParams{2, false, 128, 256, 4, 2, 64}));

// Shared-expert variant of smoke_batched_gemv_hidden_not_multiple_of_1024 (REDUCE_COUNT = MAX_TOPK + 1).
INSTANTIATE_TEST_SUITE_P(smoke_batched_gemv_hidden_not_multiple_of_1024_shared,
                         moe_3gemm_compressed_gpu_shared_random,
                         ::testing::Values(Moe3GemmTestParams{1, true, 2816, 512, 4, 2, 128},
                                           Moe3GemmTestParams{2, true, 2816, 512, 4, 2, 128}));

TEST_P(moe_3gemm_compressed_gpu_u4, moe_accuracy_test_u4) {
    auto routing_type = GetParam();
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "No immad support";
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

    auto create_f16_tensor_3d = [&](const std::vector<ov::float16>& values, int64_t d0, int64_t d1, int64_t d2) {
        auto mem = engine.allocate_memory(layout{ov::PartialShape{d0, d1, d2}, data_types::f16, format::bfyx});
        set_values(mem, values);
        get_test_stream().finish();
        return mem;
    };

    auto hidden_states = create_f16_tensor_3d(hidden_states_data, batch_size, seq_len, hidden_size);

    // Input 1: routing_weights [batch_size, seq_len, num_experts]
    auto routing_weights = create_f16_tensor_3d(router_weights_data, batch_size, seq_len, num_experts);
    auto routing_bias = create_f16_tensor(routing_bias_data, 1, 1, 1, num_experts);

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
    cldnn::MOECompressed::Config config;
    config.hidden_size = hidden_size;
    config.inter_size = inter_size;
    config.num_expert = num_experts;
    config.top_k = top_k;
    config.group_size = group_size;
    config.out_type = data_types::f16;
    config.has_zp = true;

    // Create MoERouterFused primitive
    MoERouterFused::Config router_config;
    router_config.num_expert = num_experts;
    router_config.top_k = top_k;
    std::vector<input_info> router_inputs{input_info("routing_weights")};
    if (routing_type == cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS) {
        router_config.routing_type = MoERouterFused::RoutingType::SIGMOID_BIAS;
        topology.add(data("routing_bias", routing_bias));
        auto routing_eps_mem = engine.allocate_memory({data_types::f16, format::bfyx, {1, 1, 1, 1}});
        set_values(routing_eps_mem, {ov::float16(1e-6f)});
        get_test_stream().finish();
        topology.add(data("routing_eps", routing_eps_mem));
        router_inputs.push_back(input_info("routing_bias"));
        router_inputs.push_back(input_info("routing_eps"));
    }
    topology.add(moe_router_fused("router", router_inputs, router_config));

    std::vector<input_info> moe_inputs{input_info("hidden_states"),
                                       input_info("router", 0),  // topk_weights
                                       input_info("router", 1),  // topk_indices
                                       input_info("w0_weight"),
                                       input_info("w0_scale"),
                                       input_info("w0_zp"),
                                       input_info("w1_weight"),
                                       input_info("w1_scale"),
                                       input_info("w1_zp"),
                                       input_info("w2_weight"),
                                       input_info("w2_scale"),
                                       input_info("w2_zp")};

    // Create MOECompressed primitive
    topology.add(moe_3gemm_fused_compressed("moe_3gemm_fused_compressed", moe_inputs, config));

    // Create and execute network
    auto net_config = get_test_default_config(engine);
    net_config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, net_config);
    network.set_input_data("hidden_states", hidden_states);
    network.set_input_data("routing_weights", routing_weights);

    // Verify output shape should be [batch_size, seq_len, hidden_size]
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "moe_3gemm_fused_compressed");

    auto output_prim = outputs.begin()->second.get_memory();
    get_test_stream().flush();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_prim, get_test_stream());

    auto output_layout = output_prim->get_layout();
    EXPECT_EQ(output_layout.batch(), batch_size);
    EXPECT_EQ(output_layout.feature(), seq_len);

    const auto& output_reference = routing_type == cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS ? output_ref_sigmoid_bias : output_ref_softmax;
    for (size_t i = 0; i < batch_size * seq_len * hidden_size; ++i) {
        EXPECT_NEAR(static_cast<float>(output_ptr[i]), static_cast<float>(output_reference[i]), 1e-3f);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         moe_3gemm_compressed_gpu_u4,
                         ::testing::Values(cldnn::MoERouterFused::RoutingType::SOFTMAX, cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS));

// Symmetric quantization tests (i4/i8 weights, no zero points)
class moe_3gemm_compressed_gpu_symmetric_random : public ::testing::TestWithParam<std::tuple<cldnn::MoERouterFused::RoutingType, Moe3GemmTestParams>> {
};

TEST_P(moe_3gemm_compressed_gpu_symmetric_random, moe_accuracy_test_symmetric) {
    const auto& [routing_type, param] = GetParam();
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "No immad support";
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
    config.is_signed = true;

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

    // Quantize symmetrically (no zero points)
    auto [w0_q, w0_scale] = ref.quantize_symmetric(w0_data, config.num_experts, config.hidden_size, config.inter_size, config.group_size);
    auto [w1_q, w1_scale] = ref.quantize_symmetric(w1_data, config.num_experts, config.hidden_size, config.inter_size, config.group_size);
    auto [w2_q, w2_scale] = ref.quantize_symmetric(w2_data, config.num_experts, config.inter_size, config.hidden_size, config.group_size);

    auto w0_q_packed = ref.pack(w0_q);
    auto w1_q_packed = ref.pack(w1_q);
    auto w2_q_packed = ref.pack(w2_q);

    // Create tensors
    auto weight_dt = config.is_u4 ? data_types::i4 : data_types::i8;

    auto create_weight_tensor = [&](const std::vector<uint8_t>& values, int64_t b, int64_t f, int64_t y, int64_t x) {
        auto mem = engine.allocate_memory({weight_dt, format::bfyx, {b, f, y, x}});
        set_values(mem, values);
        get_test_stream().finish();
        return mem;
    };

    // Symmetric: ZP is element::dynamic placeholder (matches real model where the transformation
    // creates Constant(element::dynamic, Shape{0}) for absent ZP in symmetric quantization).
    auto create_dynamic_zp_placeholder = [&]() {
        auto one_byte_mem = engine.allocate_memory(cldnn::layout(ov::PartialShape{1}, data_types::u8, format::bfyx), false);
        return engine.reinterpret_buffer(*one_byte_mem, cldnn::layout(ov::PartialShape{0}, data_types::dynamic, format::bfyx));
    };
    auto make_scale_sym = [&](const std::vector<ov::float16>& v, int64_t E, int64_t ofm, int64_t G) {
        auto fmt = G > 1 ? format::byfx : format::bfyx;
        auto shape = G > 1 ? ov::PartialShape{E, ofm, G, 1} : ov::PartialShape{E, ofm, 1};
        auto mem = engine.allocate_memory(layout{shape, data_types::f16, fmt});
        set_values(mem, v);
        get_test_stream().finish();
        return mem;
    };

    auto create_f16_tensor = [&](const std::vector<ov::float16>& values, int64_t b, int64_t f, int64_t y, int64_t x) {
        auto mem = engine.allocate_memory({data_types::f16, format::bfyx, {b, f, y, x}});
        set_values(mem, values);
        get_test_stream().finish();
        return mem;
    };

    auto create_f16_tensor_3d = [&](const std::vector<ov::float16>& values, int64_t d0, int64_t d1, int64_t d2) {
        auto mem = engine.allocate_memory(layout{ov::PartialShape{d0, d1, d2}, data_types::f16, format::bfyx});
        set_values(mem, values);
        get_test_stream().finish();
        return mem;
    };

    auto hidden_states_mem = create_f16_tensor_3d(hidden_states, config.batch_size, config.seq_len, config.hidden_size);
    auto routing_weights_mem = create_f16_tensor_3d(routing_weights, config.batch_size, config.seq_len, config.num_experts);
    auto routing_bias_data = rg.generate_random_1d<ov::float16>(config.num_experts, -0.5f, 0.5f, 1000);
    auto routing_bias_mem = create_f16_tensor(routing_bias_data, 1, 1, 1, config.num_experts);
    ov::float16 routing_eps_val = ov::float16(1e-6f);

    size_t group_num = config.hidden_size / config.group_size;
    size_t group_num2 = config.inter_size / config.group_size;

    auto w0_weight_mem = create_weight_tensor(w0_q_packed, config.num_experts, config.inter_size, config.group_size, group_num);
    auto w0_scale_mem = make_scale_sym(w0_scale, config.num_experts, config.inter_size, group_num);
    auto w0_zp_mem = create_dynamic_zp_placeholder();

    auto w1_weight_mem = create_weight_tensor(w1_q_packed, config.num_experts, config.inter_size, config.group_size, group_num);
    auto w1_scale_mem = make_scale_sym(w1_scale, config.num_experts, config.inter_size, group_num);
    auto w1_zp_mem = create_dynamic_zp_placeholder();

    auto w2_weight_mem = create_weight_tensor(w2_q_packed, config.num_experts, config.hidden_size, config.group_size, group_num2);
    auto w2_scale_mem = make_scale_sym(w2_scale, config.num_experts, config.hidden_size, group_num2);
    auto w2_zp_mem = create_dynamic_zp_placeholder();

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

    // Create MoERouterFused primitive
    MoERouterFused::Config router_config;
    router_config.num_expert = config.num_experts;
    router_config.top_k = config.top_k;
    std::vector<input_info> router_inputs{input_info("routing_weights")};
    if (routing_type == cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS) {
        router_config.routing_type = MoERouterFused::RoutingType::SIGMOID_BIAS;
        topology.add(data("routing_bias", routing_bias_mem));
        auto routing_eps_mem = engine.allocate_memory({data_types::f16, format::bfyx, {1, 1, 1, 1}});
        set_values(routing_eps_mem, {routing_eps_val});
        get_test_stream().finish();
        topology.add(data("routing_eps", routing_eps_mem));
        router_inputs.push_back(input_info("routing_bias"));
        router_inputs.push_back(input_info("routing_eps"));
    }
    topology.add(moe_router_fused("router", router_inputs, router_config));

    cldnn::MOECompressed::Config moe_config;
    moe_config.hidden_size = config.hidden_size;
    moe_config.inter_size = config.inter_size;
    moe_config.num_expert = config.num_experts;
    moe_config.top_k = config.top_k;
    moe_config.group_size = config.group_size;
    moe_config.out_type = data_types::f16;
    moe_config.has_zp = false;

    std::vector<input_info> moe_inputs{input_info("hidden_states"),
                                       input_info("router", 0),  // topk_weights
                                       input_info("router", 1),  // topk_indices
                                       input_info("w0_weight"),
                                       input_info("w0_scale"),
                                       input_info("w0_zp"),
                                       input_info("w1_weight"),
                                       input_info("w1_scale"),
                                       input_info("w1_zp"),
                                       input_info("w2_weight"),
                                       input_info("w2_scale"),
                                       input_info("w2_zp")};

    topology.add(moe_3gemm_fused_compressed("moe_3gemm_fused_compressed", moe_inputs, moe_config));

    auto net_config = get_test_default_config(engine);
    net_config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, net_config);
    network.set_input_data("hidden_states", hidden_states_mem);
    network.set_input_data("routing_weights", routing_weights_mem);

    auto outputs = network.execute();
    auto output_prim = outputs.begin()->second.get_memory();
    get_test_stream().flush();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_prim, get_test_stream());

    auto ref_output = routing_type == cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS
                          ? ref.run_reference_sigmoid(hidden_states, routing_weights, routing_bias_data, routing_eps_val, w0_data, w1_data, w2_data)
                          : ref.run_reference_softmax(hidden_states, routing_weights, w0_data, w1_data, w2_data);

    const float base_tolerance = routing_type == cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS ? 0.25f : 0.15f;
    const float tolerance = base_tolerance * std::max(1, static_cast<int>(config.hidden_size / 128));
    for (size_t i = 0; i < ref_output.size(); ++i) {
        ASSERT_NEAR(static_cast<float>(output_ptr[i]), static_cast<float>(ref_output[i]), tolerance);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         moe_3gemm_compressed_gpu_symmetric_random,
                         ::testing::Combine(::testing::Values(cldnn::MoERouterFused::RoutingType::SOFTMAX,
                                                              cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS),
                                            ::testing::Values(Moe3GemmTestParams{1, true, 128, 256, 4, 2, 128, true},
                                                              Moe3GemmTestParams{16, true, 128, 256, 4, 2, 128, true},
                                                              Moe3GemmTestParams{1, false, 128, 256, 4, 2, 128, true},
                                                              Moe3GemmTestParams{16, false, 128, 256, 4, 2, 128, true},
                                                              Moe3GemmTestParams{1, true, 256, 512, 4, 2, 256, true},
                                                              Moe3GemmTestParams{1, false, 256, 512, 4, 2, 256, true},
                                                              Moe3GemmTestParams{1, true, 512, 512, 4, 2, 512, true},
                                                              Moe3GemmTestParams{1, false, 512, 512, 4, 2, 512, true},
                                                              // Sub-128 group_size for symmetric quantization.
                                                              Moe3GemmTestParams{1, true, 128, 256, 4, 2, 64, true},
                                                              Moe3GemmTestParams{1, false, 128, 256, 4, 2, 64, true})));

// Symmetric quantization test with shared expert
class moe_3gemm_compressed_gpu_shared_symmetric_random : public ::testing::TestWithParam<Moe3GemmTestParams> {};

TEST_P(moe_3gemm_compressed_gpu_shared_symmetric_random, moe_accuracy_test_shared_expert_symmetric) {
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
    config.is_signed = true;
    config.has_shared_expert = true;

    Moe3GemmReference ref(config, rg);

    // Generate random data
    auto hidden_states = rg.generate_random_1d<ov::float16>(config.batch_size * config.seq_len * config.hidden_size, -1.0f, 1.0f, 1000);
    auto routing_weights = rg.generate_random_1d<ov::float16>(config.batch_size * config.seq_len * config.num_experts, 0.0f, 1.0f, 1000);

    auto w0_data = rg.generate_random_1d<float>(config.num_experts * config.hidden_size * config.inter_size, -1.0f, 0.0f, 1000);
    auto w1_data = rg.generate_random_1d<float>(config.num_experts * config.hidden_size * config.inter_size, -1.0f, 0.0f, 1000);
    auto w2_data = rg.generate_random_1d<float>(config.num_experts * config.inter_size * config.hidden_size, -1.0f, 0.0f, 1000);

    // Shared Expert Data
    auto s_gate_data = rg.generate_random_1d<float>(config.hidden_size * config.inter_size, -1.0f, 0.0f, 1000);
    auto s_up_data = rg.generate_random_1d<float>(config.hidden_size * config.inter_size, -1.0f, 0.0f, 1000);
    auto s_down_data = rg.generate_random_1d<float>(config.inter_size * config.hidden_size, -1.0f, 0.0f, 1000);
    auto s_gate_scalar_data = rg.generate_random_1d<float>(config.hidden_size, -1.0f, 0.0f, 1000);

    for (size_t i = 0; i < config.num_experts * config.hidden_size * config.inter_size; ++i) {
        w0_data[i] /= 7.0f;
        w1_data[i] /= 11.0f;
        w2_data[i] /= 7.0f;
        if (i < config.hidden_size * config.inter_size) {
            s_gate_data[i] /= 7.0f;
            s_up_data[i] /= 11.0f;
            s_down_data[i] /= 7.0f;
        }
    }

    // Quantize symmetrically (no zero points)
    auto [w0_q, w0_scale] = ref.quantize_symmetric(w0_data, config.num_experts, config.hidden_size, config.inter_size, config.group_size);
    auto [w1_q, w1_scale] = ref.quantize_symmetric(w1_data, config.num_experts, config.hidden_size, config.inter_size, config.group_size);
    auto [w2_q, w2_scale] = ref.quantize_symmetric(w2_data, config.num_experts, config.inter_size, config.hidden_size, config.group_size);

    auto [s_gate_q, s_gate_scale] = ref.quantize_symmetric(s_gate_data, 1, config.hidden_size, config.inter_size, config.group_size);
    auto [s_up_q, s_up_scale] = ref.quantize_symmetric(s_up_data, 1, config.hidden_size, config.inter_size, config.group_size);
    auto [s_down_q, s_down_scale] = ref.quantize_symmetric(s_down_data, 1, config.inter_size, config.hidden_size, config.group_size);

    auto w0_q_packed = ref.pack(w0_q);
    auto w1_q_packed = ref.pack(w1_q);
    auto w2_q_packed = ref.pack(w2_q);
    auto s_gate_q_packed = ref.pack(s_gate_q);
    auto s_up_q_packed = ref.pack(s_up_q);
    auto s_down_q_packed = ref.pack(s_down_q);

    // Create tensors with signed weight types
    auto weight_dt = config.is_u4 ? data_types::i4 : data_types::i8;

    auto create_weight_tensor = [&](const std::vector<uint8_t>& values, int64_t b, int64_t f, int64_t y, int64_t x) {
        auto mem = engine.allocate_memory({weight_dt, format::bfyx, {b, f, y, x}});
        set_values(mem, values);
        get_test_stream().finish();
        return mem;
    };

    // Symmetric: ZP is element::dynamic placeholder (matches real model).
    auto create_dynamic_zp_placeholder = [&]() {
        auto one_byte_mem = engine.allocate_memory(cldnn::layout(ov::PartialShape{1}, data_types::u8, format::bfyx), false);
        return engine.reinterpret_buffer(*one_byte_mem, cldnn::layout(ov::PartialShape{0}, data_types::dynamic, format::bfyx));
    };
    auto make_scale_sym = [&](const std::vector<ov::float16>& v, int64_t E, int64_t ofm, int64_t G) {
        auto fmt = G > 1 ? format::byfx : format::bfyx;
        auto shape = G > 1 ? ov::PartialShape{E, ofm, G, 1} : ov::PartialShape{E, ofm, 1};
        auto mem = engine.allocate_memory(layout{shape, data_types::f16, fmt});
        set_values(mem, v);
        get_test_stream().finish();
        return mem;
    };

    auto create_f16_tensor = [&](const std::vector<ov::float16>& values, int64_t b, int64_t f, int64_t y, int64_t x) {
        auto mem = engine.allocate_memory({data_types::f16, format::bfyx, {b, f, y, x}});
        set_values(mem, values);
        get_test_stream().finish();
        return mem;
    };

    auto create_scalar_gate_tensor = [&](const std::vector<float>& values) {
        std::vector<ov::float16> fp16_values;
        for (float v : values)
            fp16_values.push_back(static_cast<ov::float16>(v));
        auto mem = engine.allocate_memory({data_types::f16, format::bfyx, {1, 1, 1, static_cast<int64_t>(values.size())}});
        set_values(mem, fp16_values);
        get_test_stream().finish();
        return mem;
    };

    auto hidden_states_mem = create_f16_tensor(hidden_states, config.batch_size, config.seq_len, config.hidden_size, 1);
    auto routing_weights_mem = create_f16_tensor(routing_weights, config.batch_size, config.seq_len, config.num_experts, 1);

    size_t group_num = config.hidden_size / config.group_size;
    size_t group_num2 = config.inter_size / config.group_size;

    // Expert Weights
    auto w0_weight_mem = create_weight_tensor(w0_q_packed, config.num_experts, config.inter_size, config.group_size, group_num);
    auto w0_scale_mem = make_scale_sym(w0_scale, config.num_experts, config.inter_size, group_num);
    auto w0_zp_mem = create_dynamic_zp_placeholder();

    auto w1_weight_mem = create_weight_tensor(w1_q_packed, config.num_experts, config.inter_size, config.group_size, group_num);
    auto w1_scale_mem = make_scale_sym(w1_scale, config.num_experts, config.inter_size, group_num);
    auto w1_zp_mem = create_dynamic_zp_placeholder();

    auto w2_weight_mem = create_weight_tensor(w2_q_packed, config.num_experts, config.hidden_size, config.group_size, group_num2);
    auto w2_scale_mem = make_scale_sym(w2_scale, config.num_experts, config.hidden_size, group_num2);
    auto w2_zp_mem = create_dynamic_zp_placeholder();

    // Shared Expert Weights
    auto s_gate_weight_mem = create_weight_tensor(s_gate_q_packed, 1, config.inter_size, config.group_size, group_num);
    auto s_gate_scale_mem = make_scale_sym(s_gate_scale, 1, config.inter_size, group_num);
    auto s_gate_zp_mem = create_dynamic_zp_placeholder();

    auto s_up_weight_mem = create_weight_tensor(s_up_q_packed, 1, config.inter_size, config.group_size, group_num);
    auto s_up_scale_mem = make_scale_sym(s_up_scale, 1, config.inter_size, group_num);
    auto s_up_zp_mem = create_dynamic_zp_placeholder();

    auto s_down_weight_mem = create_weight_tensor(s_down_q_packed, 1, config.hidden_size, config.group_size, group_num2);
    auto s_down_scale_mem = make_scale_sym(s_down_scale, 1, config.hidden_size, group_num2);
    auto s_down_zp_mem = create_dynamic_zp_placeholder();

    auto s_gate_scalar_mem = create_scalar_gate_tensor(s_gate_scalar_data);

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

    topology.add(data("s_gate_weight", s_gate_weight_mem));
    topology.add(data("s_gate_scale", s_gate_scale_mem));
    topology.add(data("s_gate_zp", s_gate_zp_mem));
    topology.add(data("s_up_weight", s_up_weight_mem));
    topology.add(data("s_up_scale", s_up_scale_mem));
    topology.add(data("s_up_zp", s_up_zp_mem));
    topology.add(data("s_down_weight", s_down_weight_mem));
    topology.add(data("s_down_scale", s_down_scale_mem));
    topology.add(data("s_down_zp", s_down_zp_mem));
    topology.add(data("s_gate_scalar", s_gate_scalar_mem));

    // Create MoERouterFused primitive (softmax for shared expert tests)
    MoERouterFused::Config router_config;
    router_config.num_expert = config.num_experts;
    router_config.top_k = config.top_k;
    router_config.routing_type = MoERouterFused::RoutingType::SOFTMAX;
    topology.add(moe_router_fused("router", {input_info("routing_weights")}, router_config));

    cldnn::MOECompressed::Config moe_config;
    moe_config.hidden_size = config.hidden_size;
    moe_config.inter_size = config.inter_size;
    moe_config.num_expert = config.num_experts;
    moe_config.top_k = config.top_k;
    moe_config.group_size = config.group_size;
    moe_config.out_type = data_types::f16;
    moe_config.num_shared_expert = 1;
    moe_config.has_zp = false;

    topology.add(moe_3gemm_fused_compressed("moe_3gemm_fused_compressed",
                                            {input_info("hidden_states"),
                                             input_info("router", 0),  // topk_weights
                                             input_info("router", 1),  // topk_indices
                                             input_info("w0_weight"),
                                             input_info("w0_scale"),
                                             input_info("w0_zp"),
                                             input_info("w1_weight"),
                                             input_info("w1_scale"),
                                             input_info("w1_zp"),
                                             input_info("w2_weight"),
                                             input_info("w2_scale"),
                                             input_info("w2_zp"),
                                             // Shared Expert Inputs (indices 12-21)
                                             input_info("s_gate_weight"),
                                             input_info("s_gate_scale"),
                                             input_info("s_gate_zp"),
                                             input_info("s_up_weight"),
                                             input_info("s_up_scale"),
                                             input_info("s_up_zp"),
                                             input_info("s_down_weight"),
                                             input_info("s_down_scale"),
                                             input_info("s_down_zp"),
                                             input_info("s_gate_scalar")},
                                            moe_config));

    auto net_config = get_test_default_config(engine);
    net_config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, net_config);
    network.set_input_data("hidden_states", hidden_states_mem);
    network.set_input_data("routing_weights", routing_weights_mem);

    auto outputs = network.execute();
    auto output_prim = outputs.begin()->second.get_memory();
    get_test_stream().flush();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_prim, get_test_stream());

    auto ref_output =
        ref.run_reference_softmax(hidden_states, routing_weights, w0_data, w1_data, w2_data, s_gate_data, s_up_data, s_down_data, s_gate_scalar_data);

    const float tolerance = 0.5f * (config.hidden_size / 128);
    for (size_t i = 0; i < ref_output.size(); ++i) {
        EXPECT_NEAR(static_cast<float>(output_ptr[i]), static_cast<float>(ref_output[i]), tolerance);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         moe_3gemm_compressed_gpu_shared_symmetric_random,
                         ::testing::Values(Moe3GemmTestParams{1, true, 128, 256, 4, 2, 128, true},
                                           Moe3GemmTestParams{16, true, 128, 256, 4, 2, 128, true},
                                           Moe3GemmTestParams{1, false, 128, 256, 4, 2, 128, true},
                                           Moe3GemmTestParams{16, false, 128, 256, 4, 2, 128, true},
                                           Moe3GemmTestParams{1, true, 256, 512, 4, 2, 256, true},
                                           Moe3GemmTestParams{1, false, 256, 512, 4, 2, 256, true},
                                           Moe3GemmTestParams{1, true, 512, 512, 4, 2, 512, true},
                                           Moe3GemmTestParams{1, false, 512, 512, 4, 2, 512, true},
                                           // Sub-128 group_size for symmetric shared expert.
                                           Moe3GemmTestParams{1, true, 128, 256, 4, 2, 64, true},
                                           Moe3GemmTestParams{1, false, 128, 256, 4, 2, 64, true},
                                           Moe3GemmTestParams{1, true, 256, 512, 4, 2, 64, true}));

// ----- get_expert_mask_from_gpu buffer overflow regression test -----
// Reproduces the bug where shape predictor flattens topk_id layout [N, top_k] → [N*top_k*1.1]
// and shape[0] is incorrectly used as token count, causing N*top_k*1.1 * top_k read overflow.
namespace {
struct expert_mask_test_result {
    std::vector<int8_t> pred_flag;
    std::vector<std::vector<int>> batch;
    std::vector<std::vector<int>> topk;
};

void build_expert_mask_cpu(const int32_t* topk_id_buf, int max_tokens, int max_topk, int max_expert_num,
                           expert_mask_test_result& result) {
    result.pred_flag.assign(max_expert_num, 0);
    result.batch.assign(max_expert_num, {});
    result.topk.assign(max_expert_num, {});
    for (int b = 0; b < max_tokens; b++) {
        auto* tok_p = &topk_id_buf[b * max_topk];
        for (int t = 0; t < max_topk; t++) {
            auto expert_no = tok_p[t];
            OPENVINO_ASSERT(expert_no >= 0 && expert_no < max_expert_num);
            result.batch[expert_no].push_back(b);
            result.topk[expert_no].push_back(t + b * max_topk);
            result.pred_flag[expert_no] = 1;
        }
    }
}
}  // namespace

TEST(moe_3gemm_expert_mask, shape_predictor_overflow_regression) {
    // Simulates Gemma4-26b MoE: 128 experts, top_k=8, 274 tokens in prefill
    constexpr int num_experts = 128;
    constexpr int top_k = 8;
    constexpr int actual_tokens = 274;
    // Shape predictor flattens [274,8]=2192 elements, applies ratio 1.1: 2411
    constexpr int predictor_shape0 = static_cast<int>(274 * 8 * 1.1);  // 2411

    // Buffer allocated by predictor: 2411 int32 elements (not 2411*8)
    std::vector<int32_t> buffer(predictor_shape0, 0);
    for (int i = 0; i < actual_tokens * top_k; i++) {
        buffer[i] = (i * 7) % num_experts;
    }

    // BUG: using predictor_shape0 as token count: reads 2411*8=19288 > buffer size 2411
    size_t buggy_read = static_cast<size_t>(predictor_shape0) * top_k;
    ASSERT_GT(buggy_read, buffer.size());

    // FIX: using actual_token_num from hidden_states: reads 274*8=2192 ≤ buffer size 2411
    size_t correct_read = static_cast<size_t>(actual_tokens) * top_k;
    ASSERT_LE(correct_read, buffer.size());

    expert_mask_test_result result;
    ASSERT_NO_THROW(build_expert_mask_cpu(buffer.data(), actual_tokens, top_k, num_experts, result));

    int total = 0;
    for (int e = 0; e < num_experts; e++)
        total += static_cast<int>(result.batch[e].size());
    ASSERT_EQ(total, actual_tokens * top_k);
}

// u3 experts: decode on the moe_3gemm_int3 batched GEMV, prefill on OneDNN grouped GEMM
struct Moe3GemmU3TestParams {
    size_t seq_len;
    size_t hidden_size;
    size_t inter_size;
    size_t num_experts;
    size_t top_k;
    size_t group_size;
    size_t batched_gemv_threshold = 32;
    bool has_zp = true;
    bool is_caching_test = false;
    bool has_shared_expert = false;
};

class moe_3gemm_compressed_gpu_u3 : public ::testing::TestWithParam<Moe3GemmU3TestParams> {};

// LSB-first bit stream: value i occupies bits [3i, 3i + 3)
static std::vector<uint8_t> pack_u3(const std::vector<uint8_t>& values) {
    std::vector<uint8_t> packed((values.size() * 3 + 7) / 8, 0);
    for (size_t i = 0; i < values.size(); ++i) {
        const size_t bit = 3 * i;
        const uint32_t v = static_cast<uint32_t>(values[i] & 7u) << (bit % 8);
        packed[bit / 8] |= static_cast<uint8_t>(v);
        if (v > 0xFF) {
            packed[bit / 8 + 1] |= static_cast<uint8_t>(v >> 8);
        }
    }
    return packed;
}

TEST_P(moe_3gemm_compressed_gpu_u3, moe_accuracy_test_u3) {
    const auto& param = GetParam();
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "No immad support";
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
    config.is_u4 = true;  // quantizes to [0, 7], the u3 range
    config.has_shared_expert = param.has_shared_expert;
    Moe3GemmReference ref(config, rg);

    auto hidden_states = rg.generate_random_1d<ov::float16>(config.seq_len * config.hidden_size, -1.0f, 1.0f, 1000);
    auto routing_weights = rg.generate_random_1d<ov::float16>(config.seq_len * config.num_experts, 0.0f, 1.0f, 1000);

    const auto hidden = static_cast<int64_t>(config.hidden_size);
    const auto gs = static_cast<int64_t>(config.group_size);
    topology topology;
    std::vector<input_info> moe_inputs{input_info("hidden_states"), input_info("router", 0), input_info("router", 1)};
    // Gate, up and down of `experts` experts, quantized to u3 and added to the topology. Returns their dequantized
    // values for the reference.
    auto add_experts = [&](const std::string& prefix, size_t experts) {
        const auto E = static_cast<int64_t>(experts);
        // Weights [E, N, K] as a u3 stream; scales and zero points logically [E, N, G], physically [E, G, N]
        auto make_weight = [&](const std::vector<uint8_t>& values, int64_t n, int64_t groups) {
            auto mem = engine.allocate_memory({data_types::u3, format::bfyx, {E, n, gs, groups}});
            set_values(mem, pack_u3(values));
            return mem;
        };
        auto make_scale = [&](const std::vector<ov::float16>& v, int64_t n, int64_t groups) {
            auto mem = engine.allocate_memory(layout{ov::PartialShape{E, n, groups, 1}, data_types::f16, format::byfx});
            set_values(mem, v);
            return mem;
        };
        auto make_zp = [&](const std::vector<uint8_t>& v, int64_t n, int64_t groups) {
            if (!param.has_zp) {
                auto one_byte_mem = engine.allocate_memory(cldnn::layout(ov::PartialShape{1}, data_types::u8, format::bfyx), false);
                return engine.reinterpret_buffer(*one_byte_mem, cldnn::layout(ov::PartialShape{0}, data_types::dynamic, format::bfyx));
            }
            auto mem = engine.allocate_memory(layout{ov::PartialShape{E, n, groups, 1}, data_types::u8, format::byfx});
            set_values(mem, v);
            return mem;
        };
        std::array<std::vector<float>, 3> qdq;
        for (size_t i = 0; i < 3; ++i) {
            const bool down = i == 2;
            const size_t k = down ? config.inter_size : config.hidden_size;
            const size_t n = down ? config.hidden_size : config.inter_size;
            const auto groups = static_cast<int64_t>(k / config.group_size);
            auto values = rg.generate_random_1d<float>(experts * k * n, -1.0f, 1.0f, 1000);
            for (auto& v : values) {
                v /= i == 1 ? 11.0f : 7.0f;
            }
            auto [q, scale, zp] = ref.quantize(values, experts, k, n, config.group_size);
            // Without a zero point the weights dequantize as q * scale
            if (!param.has_zp) {
                std::fill(zp.begin(), zp.end(), uint8_t{0});
            }
            qdq[i] = ref.dequantize(q, scale, zp, experts, k, n, config.group_size);
            const std::string name = prefix + std::to_string(i);
            topology.add(data(name + "_weight", make_weight(q, static_cast<int64_t>(n), groups)));
            topology.add(data(name + "_scale", make_scale(scale, static_cast<int64_t>(n), groups)));
            topology.add(data(name + "_zp", make_zp(zp, static_cast<int64_t>(n), groups)));
            moe_inputs.insert(moe_inputs.end(), {input_info(name + "_weight"), input_info(name + "_scale"), input_info(name + "_zp")});
        }
        return qdq;
    };
    const auto w_qdq = add_experts("w", config.num_experts);
    std::array<std::vector<float>, 3> s_qdq;
    std::vector<float> s_gate_scalar;
    if (param.has_shared_expert) {
        s_qdq = add_experts("s", 1);
        auto s_gate_scalar_f16 = rg.generate_random_1d<ov::float16>(config.hidden_size, -0.1f, 0.1f, 1000);
        s_gate_scalar.assign(s_gate_scalar_f16.begin(), s_gate_scalar_f16.end());
        auto mem = engine.allocate_memory({data_types::f16, format::bfyx, {1, 1, 1, hidden}});
        set_values(mem, s_gate_scalar_f16);
        topology.add(data("s_gate_scalar", mem));
        moe_inputs.push_back(input_info("s_gate_scalar"));
    }

    auto hidden_states_mem = engine.allocate_memory(layout{ov::PartialShape{1, static_cast<int64_t>(config.seq_len), hidden}, data_types::f16, format::bfyx});
    set_values(hidden_states_mem, hidden_states);
    auto routing_weights_mem = engine.allocate_memory(
        layout{ov::PartialShape{1, static_cast<int64_t>(config.seq_len), static_cast<int64_t>(config.num_experts)}, data_types::f16, format::bfyx});
    set_values(routing_weights_mem, routing_weights);
    get_test_stream().finish();

    topology.add(input_layout("hidden_states", hidden_states_mem->get_layout()));
    topology.add(input_layout("routing_weights", routing_weights_mem->get_layout()));

    MoERouterFused::Config router_config;
    router_config.num_expert = config.num_experts;
    router_config.top_k = config.top_k;
    topology.add(moe_router_fused("router", {input_info("routing_weights")}, router_config));

    cldnn::MOECompressed::Config moe_config;
    moe_config.hidden_size = config.hidden_size;
    moe_config.inter_size = config.inter_size;
    moe_config.num_expert = config.num_experts;
    moe_config.top_k = config.top_k;
    moe_config.group_size = config.group_size;
    moe_config.out_type = data_types::f16;
    moe_config.has_zp = param.has_zp;
    moe_config.num_shared_expert = param.has_shared_expert ? 1 : 0;
    topology.add(moe_3gemm_fused_compressed("moe", moe_inputs, moe_config));

    auto net_config = get_test_default_config(engine);
    net_config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    net_config.set_property(ov::intel_gpu::moe_batched_gemv_threshold(param.batched_gemv_threshold));
    auto network = get_network(engine, topology, net_config, get_test_stream_ptr(), param.is_caching_test);
    network->set_input_data("hidden_states", hidden_states_mem);
    network->set_input_data("routing_weights", routing_weights_mem);
    auto outputs = network->execute();
    auto output_mem = outputs.at("moe").get_memory();

    const auto moe_inst = network->get_primitive("moe");
    const auto executed_kernels = moe_inst->get_impl()->get_kernels_dump_info(*moe_inst->get_impl_params()).get_entries();
    // Threshold 0 selects the per-device default, which is at least 2 tokens
    const size_t threshold = param.batched_gemv_threshold != 0 ? param.batched_gemv_threshold : 2;
    const bool expect_gemv = param.seq_len <= threshold;
    for (const char* kernel_name : {"moe_3gemm_int3_gemv_gate_up", "moe_3gemm_int3_gemv_down"}) {
        ASSERT_EQ(executed_kernels.find(kernel_name) != std::string::npos, expect_gemv) << kernel_name << ", executed kernels: " << executed_kernels;
    }

    auto ref_output = ref.run_reference_softmax(hidden_states, routing_weights, w_qdq[0], w_qdq[1], w_qdq[2], s_qdq[0], s_qdq[1], s_qdq[2], s_gate_scalar);
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_mem, get_test_stream());
    ASSERT_EQ(output_ptr.size(), ref_output.size());
    // The batched GEMV and OneDNN compute on f16 activations and stay within ~0.2% of the output range
    float ref_max = 0.0f;
    for (const auto& v : ref_output) {
        ref_max = std::max(ref_max, std::abs(static_cast<float>(v)));
    }
    const float tolerance = 0.005f * ref_max + 0.01f;
    for (size_t i = 0; i < ref_output.size(); ++i) {
        ASSERT_NEAR(static_cast<float>(output_ptr[i]), static_cast<float>(ref_output[i]), tolerance) << "i = " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         moe_3gemm_compressed_gpu_u3,
                         ::testing::Values(
                             // decode: batched GEMV, one and several tokens
                             Moe3GemmU3TestParams{1, 512, 512, 8, 2, 128},
                             Moe3GemmU3TestParams{8, 512, 512, 8, 2, 128},
                             // prefill past the threshold: OneDNN grouped GEMM
                             Moe3GemmU3TestParams{4, 512, 512, 8, 2, 128, 1},
                             Moe3GemmU3TestParams{40, 512, 512, 4, 2, 128},
                             // 64- and 32-value weight groups, and per-channel scales and zero points
                             Moe3GemmU3TestParams{1, 512, 512, 8, 2, 64},
                             Moe3GemmU3TestParams{4, 512, 512, 8, 2, 32},
                             Moe3GemmU3TestParams{2, 512, 512, 8, 2, 512},
                             // no zero point
                             Moe3GemmU3TestParams{1, 512, 512, 8, 2, 128, 32, false},
                             Moe3GemmU3TestParams{8, 512, 512, 8, 2, 64, 32, false},
                             // default threshold (per device)
                             Moe3GemmU3TestParams{2, 512, 512, 8, 2, 128, 0},
                             // model cache; the threshold must survive the import
                             Moe3GemmU3TestParams{1, 512, 512, 8, 2, 128, 32, true, true},
                             Moe3GemmU3TestParams{4, 512, 512, 8, 2, 128, 1, true, true},
                             Moe3GemmU3TestParams{40, 512, 512, 4, 2, 128, 32, true, true},
                             // u3 shared expert: an extra GEMV slot per token (decode), OneDNN (prefill)
                             Moe3GemmU3TestParams{1, 512, 512, 8, 2, 128, 32, true, false, true},
                             Moe3GemmU3TestParams{8, 512, 512, 8, 2, 128, 32, true, false, true},
                             Moe3GemmU3TestParams{4, 512, 512, 8, 2, 64, 32, false, false, true},
                             Moe3GemmU3TestParams{40, 512, 512, 4, 2, 128, 32, true, false, true},
                             Moe3GemmU3TestParams{1, 512, 512, 8, 2, 128, 32, true, true, true}));
