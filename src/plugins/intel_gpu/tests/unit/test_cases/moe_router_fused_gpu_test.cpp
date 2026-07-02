// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/moe_router_fused.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/layout.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <numeric>
#include <vector>

#include "../../common/random_generator.hpp"
#include "../test_utils/test_utils.h"

using namespace cldnn;
using namespace ::tests;

struct MoeRouterTestParams {
    size_t num_tokens;
    size_t num_experts;
    size_t top_k;
};

class moe_router_fused_softmax_gpu : public ::testing::TestWithParam<MoeRouterTestParams> {};

TEST_P(moe_router_fused_softmax_gpu, router_accuracy_test) {
    auto param = GetParam();
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "No immad support";
    }

    tests::random_generator rg(GET_SUITE_NAME);
    const size_t num_tokens = param.num_tokens;
    const size_t num_experts = param.num_experts;
    const size_t top_k = param.top_k;

    // Generate random logits
    auto logits_data = rg.generate_random_1d<ov::float16>(num_tokens * num_experts, -2.0f, 2.0f, 1000);

    // Create input memory
    auto logits_mem = engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(num_tokens), static_cast<int64_t>(num_experts)},
                                                    data_types::f16, format::bfyx});
    set_values(logits_mem, logits_data);
    get_test_stream().finish();

    // Build topology
    topology topology;
    topology.add(input_layout("logits", logits_mem->get_layout()));

    MoERouterFused::Config config;
    config.num_expert = num_experts;
    config.top_k = top_k;
    config.routing_type = MoERouterFused::RoutingType::SOFTMAX;

    topology.add(moe_router_fused("router", {input_info("logits")}, config));
    topology.add(reorder("topk_weights", input_info("router", 0), format::bfyx, data_types::f16));
    topology.add(reorder("topk_indices", input_info("router", 1), format::bfyx, data_types::i32));

    auto net_config = get_test_default_config(engine);
    net_config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, net_config);
    network.set_input_data("logits", logits_mem);

    auto outputs = network.execute();
    get_test_stream().flush();

    auto weights_mem = outputs.at("topk_weights").get_memory();
    auto indices_mem = outputs.at("topk_indices").get_memory();

    cldnn::mem_lock<ov::float16, mem_lock_type::read> weights_ptr(weights_mem, get_test_stream());
    cldnn::mem_lock<int32_t, mem_lock_type::read> indices_ptr(indices_mem, get_test_stream());

    // Compute reference: softmax + top-k + normalize
    for (size_t t = 0; t < num_tokens; ++t) {
        // Softmax
        std::vector<float> logits(num_experts);
        float max_val = std::numeric_limits<float>::lowest();
        for (size_t e = 0; e < num_experts; ++e) {
            logits[e] = static_cast<float>(logits_data[t * num_experts + e]);
            if (logits[e] > max_val) max_val = logits[e];
        }
        float sum_exp = 0.0f;
        for (size_t e = 0; e < num_experts; ++e) {
            logits[e] = std::exp(logits[e] - max_val);
            sum_exp += logits[e];
        }
        for (size_t e = 0; e < num_experts; ++e) {
            logits[e] /= sum_exp;
        }

        // Top-k by sorting
        std::vector<std::pair<float, size_t>> expert_weights;
        for (size_t e = 0; e < num_experts; ++e) {
            expert_weights.push_back({logits[e], e});
        }
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

        // Compare
        for (size_t k = 0; k < top_k; ++k) {
            float ref_weight = expert_weights[k].first / sum_weights;
            int32_t ref_index = static_cast<int32_t>(expert_weights[k].second);

            float actual_weight = static_cast<float>(weights_ptr[t * top_k + k]);
            int32_t actual_index = indices_ptr[t * top_k + k];

            ASSERT_NEAR(actual_weight, ref_weight, 0.01f)
                << "Token " << t << ", k=" << k << " weight mismatch";
            ASSERT_EQ(actual_index, ref_index)
                << "Token " << t << ", k=" << k << " index mismatch";
        }
    }
}

class moe_router_fused_sigmoid_bias_gpu : public ::testing::TestWithParam<MoeRouterTestParams> {};

TEST_P(moe_router_fused_sigmoid_bias_gpu, router_accuracy_test) {
    auto param = GetParam();
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "No immad support";
    }

    tests::random_generator rg(GET_SUITE_NAME);
    const size_t num_tokens = param.num_tokens;
    const size_t num_experts = param.num_experts;
    const size_t top_k = param.top_k;

    // Generate random data
    auto logits_data = rg.generate_random_1d<ov::float16>(num_tokens * num_experts, -2.0f, 2.0f, 1000);
    auto bias_data = rg.generate_random_1d<ov::float16>(num_experts, -0.5f, 0.5f, 1000);
    ov::float16 eps_val = ov::float16(1e-6f);

    // Create memories
    auto logits_mem = engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(num_tokens), static_cast<int64_t>(num_experts)},
                                                    data_types::f16, format::bfyx});
    set_values(logits_mem, logits_data);
    get_test_stream().finish();

    auto bias_mem = engine.allocate_memory(layout{ov::PartialShape{1, static_cast<int64_t>(num_experts)},
                                                  data_types::f16, format::bfyx});
    set_values(bias_mem, bias_data);
    get_test_stream().finish();

    auto eps_mem = engine.allocate_memory({data_types::f16, format::bfyx, {1, 1, 1, 1}});
    set_values(eps_mem, {eps_val});
    get_test_stream().finish();

    // Build topology
    topology topology;
    topology.add(input_layout("logits", logits_mem->get_layout()));
    topology.add(data("bias", bias_mem));
    topology.add(data("eps", eps_mem));

    MoERouterFused::Config config;
    config.num_expert = num_experts;
    config.top_k = top_k;
    config.routing_type = MoERouterFused::RoutingType::SIGMOID_BIAS;

    topology.add(moe_router_fused("router", {input_info("logits"), input_info("bias"), input_info("eps")}, config));
    topology.add(reorder("topk_weights", input_info("router", 0), format::bfyx, data_types::f16));
    topology.add(reorder("topk_indices", input_info("router", 1), format::bfyx, data_types::i32));

    auto net_config = get_test_default_config(engine);
    net_config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, net_config);
    network.set_input_data("logits", logits_mem);

    auto outputs = network.execute();
    get_test_stream().flush();

    auto weights_mem = outputs.at("topk_weights").get_memory();
    auto indices_mem = outputs.at("topk_indices").get_memory();

    cldnn::mem_lock<ov::float16, mem_lock_type::read> weights_ptr(weights_mem, get_test_stream());
    cldnn::mem_lock<int32_t, mem_lock_type::read> indices_ptr(indices_mem, get_test_stream());

    // Compute reference: sigmoid + bias topk + normalize
    for (size_t t = 0; t < num_tokens; ++t) {
        // Sigmoid
        std::vector<float> sigmoid_scores(num_experts);
        for (size_t e = 0; e < num_experts; ++e) {
            float logit = static_cast<float>(logits_data[t * num_experts + e]);
            sigmoid_scores[e] = 1.0f / (1.0f + std::exp(-logit));
        }

        // Selection = sigmoid + bias
        std::vector<std::pair<float, size_t>> expert_weights;
        for (size_t e = 0; e < num_experts; ++e) {
            float score = sigmoid_scores[e] + static_cast<float>(bias_data[e]);
            expert_weights.push_back({score, e});
        }
        std::partial_sort(expert_weights.begin(),
                          expert_weights.begin() + top_k,
                          expert_weights.end(),
                          [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
                              return a.first > b.first;
                          });

        // Normalize using raw sigmoid values at top-k indices + eps
        float sum_weights = 0.0f;
        for (size_t k = 0; k < top_k; ++k)
            sum_weights += sigmoid_scores[expert_weights[k].second];
        sum_weights += static_cast<float>(eps_val);

        // Compare
        for (size_t k = 0; k < top_k; ++k) {
            float ref_weight = sigmoid_scores[expert_weights[k].second] / sum_weights;
            int32_t ref_index = static_cast<int32_t>(expert_weights[k].second);

            float actual_weight = static_cast<float>(weights_ptr[t * top_k + k]);
            int32_t actual_index = indices_ptr[t * top_k + k];

            ASSERT_NEAR(actual_weight, ref_weight, 0.02f)
                << "Token " << t << ", k=" << k << " weight mismatch";
            ASSERT_EQ(actual_index, ref_index)
                << "Token " << t << ", k=" << k << " index mismatch";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         moe_router_fused_softmax_gpu,
                         ::testing::Values(MoeRouterTestParams{1, 4, 2},
                                           MoeRouterTestParams{4, 4, 2},
                                           MoeRouterTestParams{16, 8, 2},
                                           MoeRouterTestParams{1, 8, 4}));

INSTANTIATE_TEST_SUITE_P(smoke,
                         moe_router_fused_sigmoid_bias_gpu,
                         ::testing::Values(MoeRouterTestParams{1, 4, 2},
                                           MoeRouterTestParams{4, 4, 2},
                                           MoeRouterTestParams{16, 8, 2},
                                           MoeRouterTestParams{1, 8, 4}));
