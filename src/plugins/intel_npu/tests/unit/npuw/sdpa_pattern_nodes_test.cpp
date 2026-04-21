// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>
#include <limits>

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/openvino.hpp"
#include "util.hpp"

namespace {

using ExpectedNodes = ov::npuw::util::SDPAPatternNodes;

std::string make_past_key_name(const size_t idx) {
    return ov::npuw::util::make_past_key_name(idx);
}

std::string make_past_value_name(const size_t idx) {
    return ov::npuw::util::make_past_value_name(idx);
}

std::string make_present_key_name(const size_t idx) {
    return ov::npuw::util::make_present_key_name(idx);
}

std::string make_present_value_name(const size_t idx) {
    return ov::npuw::util::make_present_value_name(idx);
}

struct ModelBuildResult {
    std::shared_ptr<ov::Model> model;
    std::vector<ExpectedNodes> expected;
};

ModelBuildResult build_sdpa_model(size_t num_sdpa, bool miss_key_concat = false, bool miss_past_key_param = false) {
    using namespace ov;

    const Shape past_shape = {1, 4, 8, 64};
    const Shape new_token_shape = {1, 4, 1, 64};
    const Shape mask_shape = {1, 1, 1, 9};

    ParameterVector params;
    ResultVector results;
    std::vector<ExpectedNodes> expected;
    expected.reserve(num_sdpa);

    std::shared_ptr<ov::Node> prev_attn_out;  // Track output of previous SDPA to chain them

    for (size_t n = 0; n < num_sdpa; ++n) {
        const std::string idx = std::to_string(n);

        auto make_param = [&](const std::string& name, const Shape& shape) {
            auto p = std::make_shared<op::v0::Parameter>(element::f32, shape);
            p->set_friendly_name(name);
            p->output(0).get_tensor().set_names({name});
            params.push_back(p);
            return p;
        };

        auto past_value = make_param(make_past_value_name(n), past_shape);
        std::shared_ptr<op::v0::Parameter> past_key;
        std::shared_ptr<op::v0::Parameter> fallback_key;
        if (miss_past_key_param) {
            fallback_key = make_param("fallback_key." + idx, past_shape);
        } else {
            past_key = make_param(make_past_key_name(n), past_shape);
        }

        // Use output from previous SDPA as query for current SDPA (chain layers)
        std::shared_ptr<ov::Node> query;
        if (n == 0) {
            query = make_param("query." + idx, new_token_shape);
        } else {
            // Reshape previous attention output to query shape and use as input
            auto reshape_shape = op::v0::Constant::create(element::i64, Shape{4},
                                                          {1, 4, 1, 64});
            query = std::make_shared<op::v1::Reshape>(prev_attn_out, reshape_shape, false);
            query->set_friendly_name("query_from_prev." + idx);
        }

        auto new_key = make_param("new_key." + idx, new_token_shape);
        auto new_value = make_param("new_value." + idx, new_token_shape);
        auto mask = make_param("mask." + idx, mask_shape);

        std::shared_ptr<ov::Node> key_path;
        std::shared_ptr<ov::Node> key_concat;
        if (miss_key_concat) {
            key_path = new_key;
        } else {
            key_concat = std::make_shared<op::v0::Concat>(OutputVector{(miss_past_key_param ? fallback_key : past_key), new_key}, 2);
            key_concat->set_friendly_name("concat_key." + idx);
            key_path = key_concat;
        }

        auto value_concat = std::make_shared<op::v0::Concat>(OutputVector{past_value, new_value}, 2);
        value_concat->set_friendly_name("concat_value." + idx);

        auto qk = std::make_shared<op::v0::MatMul>(query, key_path, false, true);
        qk->set_friendly_name("matmul1." + idx);
        auto add = std::make_shared<op::v1::Add>(qk->output(0), mask->output(0));
        add->set_friendly_name("add." + idx);
        auto softmax = std::make_shared<op::v8::Softmax>(add->output(0), 3);
        softmax->set_friendly_name("softmax." + idx);
        auto matmul2 = std::make_shared<op::v0::MatMul>(softmax->output(0), value_concat->output(0));
        matmul2->set_friendly_name("matmul2." + idx);

        auto make_result = [&](const ov::Output<ov::Node>& out, const std::string& name) {
            auto r = std::make_shared<op::v0::Result>(out);
            r->set_friendly_name(name);
            r->output(0).get_tensor().set_names({name});
            results.push_back(r);
        };

        make_result(key_path->output(0), make_present_key_name(n));
        make_result(value_concat->output(0), make_present_value_name(n));
        make_result(matmul2->output(0), "attn_out." + idx);

        prev_attn_out = matmul2;  // Store output for next iteration

        expected.push_back(ExpectedNodes{qk,
                         matmul2,
                         softmax,
                         add,
                         key_concat,
                         value_concat});
    }

    auto model = std::make_shared<Model>(results, params, "sdpa_pattern_nodes_test_model");
    model->validate_nodes_and_infer_types();
    //ov::save_model(model, "sdpa_pattern_nodes_test_model.xml");
    return {model, expected};
}

void expect_nodes_equal(const ov::npuw::util::SDPAPatternNodes& actual, const ExpectedNodes& expected) {
    EXPECT_EQ(actual.matmul1_node, expected.matmul1_node);
    EXPECT_EQ(actual.matmul2_node, expected.matmul2_node);
    EXPECT_EQ(actual.softmax_node, expected.softmax_node);
    EXPECT_EQ(actual.add_node, expected.add_node);
    // EXPECT_EQ(actual.past_key_param_node, expected.past_key_param_node);
    // EXPECT_EQ(actual.past_value_param_node, expected.past_value_param_node);
    EXPECT_EQ(actual.past_key_concat_node, expected.past_key_concat_node);
    EXPECT_EQ(actual.past_value_concat_node, expected.past_value_concat_node);
}

std::string sort_key(const ExpectedNodes& nodes) {
    return nodes.matmul1_node->get_friendly_name();
}

struct PoisonBuildResult {
    std::shared_ptr<ov::Node> chain_out;
};

PoisonBuildResult make_poisoned_sdpa(const std::string& idx,
                                     size_t poison_type,
                                     const std::shared_ptr<ov::Node>& query_in,
                                     ov::ParameterVector& params,
                                     ov::ResultVector& results) {
    using namespace ov;

    const Shape q_shape = {1, 4, 1, 64};
    const Shape k_shape = {1, 4, 9, 64};
    const Shape v_shape = {1, 4, 9, 64};
    const Shape score_shape = {1, 4, 1, 9};
    const Shape score_bias_shape = {1, 1, 1, 9};

    auto make_param = [&](const std::string& name, const Shape& shape) {
        auto p = std::make_shared<op::v0::Parameter>(element::f32, shape);
        p->set_friendly_name(name);
        p->output(0).get_tensor().set_names({name});
        params.push_back(p);
        return p;
    };

    auto q = query_in;
    if (!q) {
        q = make_param("poison_query." + idx, q_shape);
    }

    auto k = make_param("poison_key." + idx, k_shape);
    auto v = make_param("poison_value." + idx, v_shape);

    std::shared_ptr<ov::Node> poison_out;
    std::shared_ptr<ov::Node> chain_out;
    switch (poison_type % 4) {
    case 0: {
        // Missing Add: MatMul -> Softmax -> MatMul
        auto mm1 = std::make_shared<op::v0::MatMul>(q, k, false, true);
        mm1->set_friendly_name("poison_matmul1." + idx);
        auto sm = std::make_shared<op::v8::Softmax>(mm1, 3);
        sm->set_friendly_name("poison_softmax." + idx);
        auto mm2 = std::make_shared<op::v0::MatMul>(sm, v);
        mm2->set_friendly_name("poison_matmul2." + idx);
        poison_out = mm2;
        chain_out = mm2;
        break;
    }
    case 1: {
        // Missing second MatMul: MatMul -> Add -> Softmax
        auto mm1 = std::make_shared<op::v0::MatMul>(q, k, false, true);
        mm1->set_friendly_name("poison_matmul1." + idx);
        auto mask = make_param("poison_mask." + idx, score_bias_shape);
        auto add = std::make_shared<op::v1::Add>(mm1, mask);
        add->set_friendly_name("poison_add." + idx);
        auto sm = std::make_shared<op::v8::Softmax>(add, 3);
        sm->set_friendly_name("poison_softmax." + idx);
        poison_out = sm;
        // Keep the global chain shape stable for next attention block.
        chain_out = q;
        break;
    }
    case 2: {
        // Missing first MatMul: Add -> Softmax -> MatMul
        auto score = make_param("poison_score." + idx, score_shape);
        auto mask = make_param("poison_mask." + idx, score_bias_shape);
        auto add = std::make_shared<op::v1::Add>(score, mask);
        add->set_friendly_name("poison_add." + idx);
        auto sm = std::make_shared<op::v8::Softmax>(add, 3);
        sm->set_friendly_name("poison_softmax." + idx);
        auto mm2 = std::make_shared<op::v0::MatMul>(sm, v);
        mm2->set_friendly_name("poison_matmul2." + idx);
        poison_out = mm2;
        chain_out = mm2;
        break;
    }
    default: {
        // Missing Softmax: MatMul -> Add -> MatMul
        auto mm1 = std::make_shared<op::v0::MatMul>(q, k, false, true);
        mm1->set_friendly_name("poison_matmul1." + idx);
        auto mask = make_param("poison_mask." + idx, score_bias_shape);
        auto add = std::make_shared<op::v1::Add>(mm1, mask);
        add->set_friendly_name("poison_add." + idx);
        auto mm2 = std::make_shared<op::v0::MatMul>(add, v);
        mm2->set_friendly_name("poison_matmul2." + idx);
        poison_out = mm2;
        chain_out = mm2;
        break;
    }
    }

    auto r = std::make_shared<op::v0::Result>(poison_out);
    r->set_friendly_name("poison_out." + idx);
    r->output(0).get_tensor().set_names({"poison_out." + idx});
    results.push_back(r);

    return {chain_out};
}

ModelBuildResult build_noisy_sdpa_model(size_t num_sdpa, size_t broken_idx, size_t poison_count) {
    using namespace ov;

    const Shape past_shape = {1, 4, 8, 64};
    const Shape new_token_shape = {1, 4, 1, 64};
    const Shape context_shape = {1, 4, 9, 64};
    const Shape mask_shape = {1, 1, 1, 9};

    ParameterVector params;
    ResultVector results;
    std::vector<ExpectedNodes> expected;
    expected.reserve(num_sdpa);

    std::shared_ptr<ov::Node> prev_attn_out;
    size_t emitted_poison = 0;

    auto maybe_add_poison = [&]() {
        if (emitted_poison >= poison_count) {
            return;
        }
        const std::string poison_idx = std::to_string(emitted_poison);
        auto poison = make_poisoned_sdpa(poison_idx, emitted_poison, prev_attn_out, params, results);
        prev_attn_out = poison.chain_out;
        emitted_poison++;
    };

    for (size_t n = 0; n < num_sdpa; ++n) {
        const std::string idx = std::to_string(n);
        const bool is_broken = (n == broken_idx);

        auto make_param = [&](const std::string& name, const Shape& shape) {
            auto p = std::make_shared<op::v0::Parameter>(element::f32, shape);
            p->set_friendly_name(name);
            p->output(0).get_tensor().set_names({name});
            params.push_back(p);
            return p;
        };

        std::shared_ptr<op::v0::Parameter> past_key;
        std::shared_ptr<op::v0::Parameter> past_value;

        if (!is_broken) {
            past_key = make_param(make_past_key_name(n), past_shape);
            past_value = make_param(make_past_value_name(n), past_shape);
        }

        std::shared_ptr<ov::Node> query;
        if (!prev_attn_out) {
            query = make_param("query." + idx, new_token_shape);
        } else {
            query = prev_attn_out;
        }
        auto new_key = make_param("new_key." + idx, new_token_shape);
        auto new_value = make_param("new_value." + idx, new_token_shape);
        auto mask = make_param("mask." + idx, mask_shape);

        std::shared_ptr<ov::Node> key_path;
        std::shared_ptr<ov::Node> value_path;
        std::shared_ptr<ov::Node> key_concat;
        std::shared_ptr<ov::Node> value_concat;

        if (is_broken) {
            // No-concat SDPA: direct cache parameters without KV-cache concat, pattern is invalid.
            key_path = make_param("cache_key." + idx, context_shape);
            value_path = make_param("cache_value." + idx, context_shape);
        } else {
            key_concat = std::make_shared<op::v0::Concat>(OutputVector{past_key, new_key}, 2);
            key_concat->set_friendly_name("concat_key." + idx);
            key_path = key_concat;

            value_concat = std::make_shared<op::v0::Concat>(OutputVector{past_value, new_value}, 2);
            value_concat->set_friendly_name("concat_value." + idx);
            value_path = value_concat;
        }

        auto qk = std::make_shared<op::v0::MatMul>(query, key_path, false, true);
        qk->set_friendly_name("matmul1." + idx);
        auto add = std::make_shared<op::v1::Add>(qk->output(0), mask->output(0));
        add->set_friendly_name("add." + idx);
        auto softmax = std::make_shared<op::v8::Softmax>(add->output(0), 3);
        softmax->set_friendly_name("softmax." + idx);
        auto matmul2 = std::make_shared<op::v0::MatMul>(softmax->output(0), value_path->output(0));
        matmul2->set_friendly_name("matmul2." + idx);

        auto make_result = [&](const ov::Output<ov::Node>& out, const std::string& name) {
            auto r = std::make_shared<op::v0::Result>(out);
            r->set_friendly_name(name);
            r->output(0).get_tensor().set_names({name});
            results.push_back(r);
        };

        make_result(key_path->output(0), make_present_key_name(n));
        make_result(value_path->output(0), make_present_value_name(n));
        make_result(matmul2->output(0), "attn_out." + idx);

        prev_attn_out = matmul2;

        expected.push_back(ExpectedNodes{qk,
                         matmul2,
                         softmax,
                         add,
                         key_concat,
                         value_concat});

        maybe_add_poison();
    }

    while (emitted_poison < poison_count) {
        maybe_add_poison();
    }

    auto model = std::make_shared<Model>(results, params, "sdpa_pattern_nodes_noisy_test_model");
    model->validate_nodes_and_infer_types();
    return {model, expected};
}

ModelBuildResult build_sdpa_model_with_wrapped_concats(size_t num_sdpa) {
    using namespace ov;

    const Shape past_shape = {1, 4, 8, 64};
    const Shape new_token_shape = {1, 4, 1, 64};
    const Shape cache_shape = {1, 4, 9, 64};
    const Shape mask_shape = {1, 1, 1, 9};

    ParameterVector params;
    ResultVector results;
    std::vector<ExpectedNodes> expected;
    expected.reserve(num_sdpa);

    for (size_t n = 0; n < num_sdpa; ++n) {
        const std::string idx = std::to_string(n);

        auto make_param = [&](const std::string& name, const Shape& shape) {
            auto p = std::make_shared<op::v0::Parameter>(element::f32, shape);
            p->set_friendly_name(name);
            p->output(0).get_tensor().set_names({name});
            params.push_back(p);
            return p;
        };

        auto past_key = make_param(make_past_key_name(n), past_shape);
        auto past_value = make_param(make_past_value_name(n), past_shape);
        auto query = make_param("query_wrapped." + idx, new_token_shape);
        auto new_key = make_param("new_key_wrapped." + idx, new_token_shape);
        auto new_value = make_param("new_value_wrapped." + idx, new_token_shape);
        auto mask = make_param("mask_wrapped." + idx, mask_shape);

        auto key_concat = std::make_shared<op::v0::Concat>(OutputVector{past_key, new_key}, 2);
        key_concat->set_friendly_name("concat_key_wrapped." + idx);
        auto value_concat = std::make_shared<op::v0::Concat>(OutputVector{past_value, new_value}, 2);
        value_concat->set_friendly_name("concat_value_wrapped." + idx);

        auto shape_c = op::v0::Constant::create(element::i64, Shape{4},
                                                {static_cast<int64_t>(cache_shape[0]),
                                                 static_cast<int64_t>(cache_shape[1]),
                                                 static_cast<int64_t>(cache_shape[2]),
                                                 static_cast<int64_t>(cache_shape[3])});

        auto key_reshape = std::make_shared<op::v1::Reshape>(key_concat, shape_c, false);
        key_reshape->set_friendly_name("reshape_key_wrapped." + idx);
        auto value_reshape = std::make_shared<op::v1::Reshape>(value_concat, shape_c, false);
        value_reshape->set_friendly_name("reshape_value_wrapped." + idx);

        auto qk = std::make_shared<op::v0::MatMul>(query, key_reshape, false, true);
        qk->set_friendly_name("matmul1_wrapped." + idx);
        auto add = std::make_shared<op::v1::Add>(qk->output(0), mask->output(0));
        add->set_friendly_name("add_wrapped." + idx);
        auto softmax = std::make_shared<op::v8::Softmax>(add->output(0), 3);
        softmax->set_friendly_name("softmax_wrapped." + idx);
        auto matmul2 = std::make_shared<op::v0::MatMul>(softmax->output(0), value_reshape->output(0));
        matmul2->set_friendly_name("matmul2_wrapped." + idx);

        auto make_result = [&](const ov::Output<ov::Node>& out, const std::string& name) {
            auto r = std::make_shared<op::v0::Result>(out);
            r->set_friendly_name(name);
            r->output(0).get_tensor().set_names({name});
            results.push_back(r);
        };

        make_result(key_reshape->output(0), make_present_key_name(n));
        make_result(value_reshape->output(0), make_present_value_name(n));
        make_result(matmul2->output(0), "attn_out_wrapped." + idx);

        expected.push_back(ExpectedNodes{qk,
                         matmul2,
                         softmax,
                         add,
                         key_concat,
                         value_concat});
    }

    auto model = std::make_shared<Model>(results, params, "sdpa_pattern_nodes_wrapped_concat_model");
    model->validate_nodes_and_infer_types();
    return {model, expected};
}


ModelBuildResult build_sdpa_model_without_concats(size_t num_sdpa) {
    using namespace ov;

    const Shape cache_shape = {1, 4, 9, 64};
    const Shape query_shape = {1, 4, 1, 64};
    const Shape mask_shape = {1, 1, 1, 9};

    ParameterVector params;
    ResultVector results;
    std::vector<ExpectedNodes> expected;
    expected.reserve(num_sdpa);

    for (size_t n = 0; n < num_sdpa; ++n) {
        const std::string idx = std::to_string(n);

        auto make_param = [&](const std::string& name, const Shape& shape) {
            auto p = std::make_shared<op::v0::Parameter>(element::f32, shape);
            p->set_friendly_name(name);
            p->output(0).get_tensor().set_names({name});
            params.push_back(p);
            return p;
        };

        auto query = make_param("query_noconcat." + idx, query_shape);
        auto key_context = make_param("cache_key_noconcat." + idx, cache_shape);
        auto value_context = make_param("cache_value_noconcat." + idx, cache_shape);
        auto mask = make_param("mask_noconcat." + idx, mask_shape);

        auto qk = std::make_shared<op::v0::MatMul>(query, key_context, false, true);
        qk->set_friendly_name("matmul1_noconcat." + idx);
        auto add = std::make_shared<op::v1::Add>(qk->output(0), mask->output(0));
        add->set_friendly_name("add_noconcat." + idx);
        auto softmax = std::make_shared<op::v8::Softmax>(add->output(0), 3);
        softmax->set_friendly_name("softmax_noconcat." + idx);
        auto matmul2 = std::make_shared<op::v0::MatMul>(softmax->output(0), value_context->output(0));
        matmul2->set_friendly_name("matmul2_noconcat." + idx);

        auto out = std::make_shared<op::v0::Result>(matmul2);
        out->set_friendly_name("attn_out_noconcat." + idx);
        out->output(0).get_tensor().set_names({"attn_out_noconcat." + idx});
        results.push_back(out);

        expected.push_back(ExpectedNodes{qk, matmul2, softmax, add, nullptr, nullptr});
    }

    auto model = std::make_shared<Model>(results, params, "sdpa_pattern_nodes_no_concat_model");
    model->validate_nodes_and_infer_types();
    return {model, expected};
}

TEST(SdpaPatternNodesTest, MissingConcatKeepsSearchStableAndReturnsInvalidPattern) {
    const auto built = build_sdpa_model(/*num_sdpa=*/1, /*miss_key_concat=*/true, /*miss_past_key_param=*/false);

    auto pattern = ov::npuw::util::find_sdpa_pattern_nodes(built.model);
    const auto& expected = built.expected.front();

    EXPECT_FALSE(pattern.is_valid());
    EXPECT_EQ(pattern.matmul1_node, expected.matmul1_node);
    EXPECT_EQ(pattern.matmul2_node, expected.matmul2_node);
    EXPECT_EQ(pattern.softmax_node, expected.softmax_node);
    EXPECT_EQ(pattern.add_node, expected.add_node);
    // EXPECT_EQ(pattern.past_key_param_node, expected.past_key_param_node);
    // EXPECT_EQ(pattern.past_value_param_node, expected.past_value_param_node);
    EXPECT_EQ(pattern.past_key_concat_node, nullptr);
    EXPECT_EQ(pattern.past_value_concat_node, expected.past_value_concat_node);
}

TEST(SdpaPatternNodesTest, SingleBrokenSdpaWithoutAnyConcatReturnsInvalidPattern) {
    const auto built = build_noisy_sdpa_model(/*num_sdpa=*/1, /*broken_idx=*/0, /*poison_count=*/0);

    auto pattern = ov::npuw::util::find_sdpa_pattern_nodes(built.model);
    const auto& expected = built.expected.front();

    EXPECT_FALSE(pattern.is_valid());
    EXPECT_EQ(pattern.matmul1_node, expected.matmul1_node);
    EXPECT_EQ(pattern.matmul2_node, expected.matmul2_node);
    EXPECT_EQ(pattern.softmax_node, expected.softmax_node);
    EXPECT_EQ(pattern.add_node, expected.add_node);
    // EXPECT_EQ(pattern.past_key_param_node, expected.past_key_param_node);
    // EXPECT_EQ(pattern.past_value_param_node, expected.past_value_param_node);
    EXPECT_EQ(pattern.past_key_concat_node, nullptr);
    EXPECT_EQ(pattern.past_value_concat_node, nullptr);
}

TEST(SdpaPatternNodesTest, SingleCompletedSdpaReturnsExpectedNodeSet) {
    const auto built = build_sdpa_model(/*num_sdpa=*/1, /*miss_key_concat=*/false, /*miss_past_key_param=*/false);

    auto pattern = ov::npuw::util::find_sdpa_pattern_nodes(built.model);

    ASSERT_TRUE(pattern.is_valid());
    expect_nodes_equal(pattern, built.expected.front());
}

TEST(SdpaPatternNodesTest, SingleModeReturnsFirstPatternWhenMultipleSdpasExist) {
    const auto built = build_sdpa_model(/*num_sdpa=*/2, /*miss_key_concat=*/false, /*miss_past_key_param=*/false);

    auto pattern = ov::npuw::util::find_sdpa_pattern_nodes(built.model);

    ASSERT_TRUE(pattern.is_valid());
    expect_nodes_equal(pattern, built.expected.front());
}


TEST(SdpaPatternNodesTest, FindAllReturnsAllSdpaPatternsWithExactNodes) {
    const auto built = build_sdpa_model(/*num_sdpa=*/2, /*miss_key_concat=*/false, /*miss_past_key_param=*/false);

    auto patterns = ov::npuw::util::find_all_sdpa_pattern_nodes(built.model);
    auto expected = built.expected;

    ASSERT_EQ(patterns.size(), built.expected.size());
    ASSERT_EQ(patterns.size(), 2u);

    std::sort(patterns.begin(), patterns.end(), [](const auto& lhs, const auto& rhs) {
        return sort_key(lhs) < sort_key(rhs);
    });
    std::sort(expected.begin(), expected.end(), [](const auto& lhs, const auto& rhs) {
        return sort_key(lhs) < sort_key(rhs);
    });

    for (size_t i = 0; i < patterns.size(); ++i) {
        ASSERT_TRUE(patterns[i].is_valid()) << "Pattern at index " << i << " must be valid";
        expect_nodes_equal(patterns[i], expected[i]);
    }
}

TEST(SdpaPatternNodesTest, FindAllIgnoresPoisonSubgraphsInLargePrefillLikeModel) {
    const auto built = build_noisy_sdpa_model(/*num_sdpa=*/10, /*broken_idx=*/std::numeric_limits<size_t>::max(),
                                              /*poison_count=*/7);

    auto patterns = ov::npuw::util::find_all_sdpa_pattern_nodes(built.model);
    auto expected = built.expected;

    ASSERT_EQ(patterns.size(), expected.size());
    ASSERT_EQ(patterns.size(), 10u);

    std::sort(patterns.begin(), patterns.end(), [](const auto& lhs, const auto& rhs) {
        return sort_key(lhs) < sort_key(rhs);
    });
    std::sort(expected.begin(), expected.end(), [](const auto& lhs, const auto& rhs) {
        return sort_key(lhs) < sort_key(rhs);
    });

    for (size_t i = 0; i < patterns.size(); ++i) {
        ASSERT_TRUE(patterns[i].is_valid()) << "Pattern at index " << i << " must be valid";
        expect_nodes_equal(patterns[i], expected[i]);
    }
}

TEST(SdpaPatternNodesTest, FindAllSkipsBrokenSdpaAndFindsRemainingInLargePrefillLikeModel) {
    const auto built = build_noisy_sdpa_model(/*num_sdpa=*/10, /*broken_idx=*/4, /*poison_count=*/7);

    auto patterns = ov::npuw::util::find_all_sdpa_pattern_nodes(built.model);
    auto expected = built.expected;
    expected.erase(std::remove_if(expected.begin(), expected.end(), [](const auto& nodes) {
                       return !nodes.is_valid();
                   }),
                   expected.end());

    ASSERT_EQ(patterns.size(), 10u);

    auto valid_patterns = patterns;
    valid_patterns.erase(std::remove_if(valid_patterns.begin(), valid_patterns.end(), [](const auto& nodes) {
                    return !nodes.is_valid();
                }),
                valid_patterns.end());


    ASSERT_EQ(valid_patterns.size(), 9u);

    std::sort(valid_patterns.begin(), valid_patterns.end(), [](const auto& lhs, const auto& rhs) {
        return sort_key(lhs) < sort_key(rhs);
    });
    std::sort(expected.begin(), expected.end(), [](const auto& lhs, const auto& rhs) {
        return sort_key(lhs) < sort_key(rhs);
    });

    for (size_t i = 0; i < valid_patterns.size(); ++i) {
        ASSERT_TRUE(valid_patterns[i].is_valid()) << "Pattern at index " << i << " must be valid";
        expect_nodes_equal(valid_patterns[i], expected[i]);
    }
}

TEST(SdpaPatternNodesTest, FindAllHandlesConcatWrappedByReshape) {
    const auto built = build_sdpa_model_with_wrapped_concats(/*num_sdpa=*/3);

    auto patterns = ov::npuw::util::find_all_sdpa_pattern_nodes(built.model);
    auto expected = built.expected;

    ASSERT_EQ(patterns.size(), expected.size());
    ASSERT_EQ(patterns.size(), 3u);

    std::sort(patterns.begin(), patterns.end(), [](const auto& lhs, const auto& rhs) {
        return sort_key(lhs) < sort_key(rhs);
    });
    std::sort(expected.begin(), expected.end(), [](const auto& lhs, const auto& rhs) {
        return sort_key(lhs) < sort_key(rhs);
    });

    for (size_t i = 0; i < patterns.size(); ++i) {
        ASSERT_TRUE(patterns[i].is_valid()) << "Pattern at index " << i << " must be valid";
        expect_nodes_equal(patterns[i], expected[i]);
    }
}


TEST(SdpaPatternNodesTest, FindAllReturnsAllSdpaWhenNoConcatIsPresent) {
    const auto built = build_sdpa_model_without_concats(/*num_sdpa=*/4);

    auto patterns = ov::npuw::util::find_all_sdpa_pattern_nodes(built.model);
    auto expected = built.expected;

    ASSERT_EQ(patterns.size(), expected.size());
    ASSERT_EQ(patterns.size(), 4u);

    std::sort(patterns.begin(), patterns.end(), [](const auto& lhs, const auto& rhs) {
        return sort_key(lhs) < sort_key(rhs);
    });
    std::sort(expected.begin(), expected.end(), [](const auto& lhs, const auto& rhs) {
        return sort_key(lhs) < sort_key(rhs);
    });

    for (size_t i = 0; i < patterns.size(); ++i) {
        ASSERT_FALSE(patterns[i].is_valid()) << "Pattern at index " << i << " is expected to be invalid without concats";
        expect_nodes_equal(patterns[i], expected[i]);
    }
}

}  // namespace
