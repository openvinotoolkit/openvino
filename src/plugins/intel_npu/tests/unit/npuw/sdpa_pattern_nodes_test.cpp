// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

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

struct SdpaNodes {
    std::shared_ptr<ov::Node> matmul1;
    std::shared_ptr<ov::Node> matmul2;
    std::shared_ptr<ov::Node> softmax;
    std::shared_ptr<ov::Node> add;
    std::shared_ptr<ov::Node> past_key_param;
    std::shared_ptr<ov::Node> past_value_param;
    std::shared_ptr<ov::Node> past_key_concat;
    std::shared_ptr<ov::Node> past_value_concat;
};

struct ModelBuildResult {
    std::shared_ptr<ov::Model> model;
    std::vector<SdpaNodes> expected;
};

ModelBuildResult build_sdpa_model(size_t num_sdpa, bool miss_key_concat = false, bool miss_past_key_param = false) {
    using namespace ov;

    const Shape past_shape = {1, 4, 8, 64};
    const Shape new_token_shape = {1, 4, 1, 64};
    const Shape mask_shape = {1, 1, 1, 9};

    ParameterVector params;
    ResultVector results;
    std::vector<SdpaNodes> expected;
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

        auto past_value = make_param("past_key_values." + idx + ".value", past_shape);
        std::shared_ptr<op::v0::Parameter> past_key;
        std::shared_ptr<op::v0::Parameter> fallback_key;
        if (miss_past_key_param) {
            fallback_key = make_param("fallback_key." + idx, past_shape);
        } else {
            past_key = make_param("past_key_values." + idx + ".key", past_shape);
        }

        auto query = make_param("query." + idx, new_token_shape);
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

        make_result(key_path->output(0), "present." + idx + ".key");
        make_result(value_concat->output(0), "present." + idx + ".value");
        make_result(matmul2->output(0), "attn_out." + idx);

        expected.push_back(SdpaNodes{qk,
                                     matmul2,
                                     softmax,
                                     add,
                                     past_key,
                                     past_value,
                                     key_concat,
                                     value_concat});
    }

    auto model = std::make_shared<Model>(results, params, "sdpa_pattern_nodes_test_model");
    model->validate_nodes_and_infer_types();
    return {model, expected};
}

void expect_nodes_equal(const ov::npuw::util::SDPAPatternNodes& actual, const SdpaNodes& expected) {
    EXPECT_EQ(actual.matmul1_node, expected.matmul1);
    EXPECT_EQ(actual.matmul2_node, expected.matmul2);
    EXPECT_EQ(actual.softmax_node, expected.softmax);
    EXPECT_EQ(actual.add_node, expected.add);
    EXPECT_EQ(actual.past_key_param_node, expected.past_key_param);
    EXPECT_EQ(actual.past_value_param_node, expected.past_value_param);
    EXPECT_EQ(actual.past_key_concat_node, expected.past_key_concat);
    EXPECT_EQ(actual.past_value_concat_node, expected.past_value_concat);
}

std::string sort_key(const SdpaNodes& nodes) {
    return nodes.matmul1->get_friendly_name();
}

std::string sort_key(const ov::npuw::util::SDPAPatternNodes& nodes) {
    return nodes.matmul1_node->get_friendly_name();
}

void add_poison_subgraphs(ov::ParameterVector& params, ov::ResultVector& results, size_t poison_count) {
    using namespace ov;
    const Shape q_shape = {1, 4, 1, 64};
    const Shape k_shape = {1, 4, 9, 64};
    const Shape v_shape = {1, 4, 9, 64};

    for (size_t i = 0; i < poison_count; ++i) {
        const std::string idx = std::to_string(i);

        auto q = std::make_shared<op::v0::Parameter>(element::f32, q_shape);
        q->set_friendly_name("poison_query." + idx);
        q->output(0).get_tensor().set_names({"poison_query." + idx});
        params.push_back(q);

        auto k = std::make_shared<op::v0::Parameter>(element::f32, k_shape);
        k->set_friendly_name("poison_key." + idx);
        k->output(0).get_tensor().set_names({"poison_key." + idx});
        params.push_back(k);

        auto v = std::make_shared<op::v0::Parameter>(element::f32, v_shape);
        v->set_friendly_name("poison_value." + idx);
        v->output(0).get_tensor().set_names({"poison_value." + idx});
        params.push_back(v);

        auto mm1 = std::make_shared<op::v0::MatMul>(q, k, false, true);
        mm1->set_friendly_name("poison_matmul1." + idx);
        auto sm = std::make_shared<op::v8::Softmax>(mm1, 3);
        sm->set_friendly_name("poison_softmax." + idx);
        auto mm2 = std::make_shared<op::v0::MatMul>(sm, v);
        mm2->set_friendly_name("poison_matmul2." + idx);

        auto r = std::make_shared<op::v0::Result>(mm2);
        r->set_friendly_name("poison_out." + idx);
        r->output(0).get_tensor().set_names({"poison_out." + idx});
        results.push_back(r);
    }
}

ModelBuildResult build_noisy_sdpa_model(size_t num_sdpa, size_t broken_idx, size_t poison_count) {
    using namespace ov;

    const Shape past_shape = {1, 4, 8, 64};
    const Shape new_token_shape = {1, 4, 1, 64};
    const Shape context_shape = {1, 4, 9, 64};
    const Shape mask_shape = {1, 1, 1, 9};

    ParameterVector params;
    ResultVector results;
    std::vector<SdpaNodes> expected;
    expected.reserve(num_sdpa > 0 ? num_sdpa - 1 : 0);

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
        std::shared_ptr<op::v0::Parameter> fallback_key;
        std::shared_ptr<op::v0::Parameter> fallback_value;

        if (is_broken) {
            fallback_key = make_param("broken_fallback_key." + idx, past_shape);
            fallback_value = make_param("broken_fallback_value." + idx, past_shape);
        } else {
            past_key = make_param("past_key_values." + idx + ".key", past_shape);
            past_value = make_param("past_key_values." + idx + ".value", past_shape);
        }

        auto query = make_param("query." + idx, new_token_shape);
        auto new_key = make_param("new_key." + idx, new_token_shape);
        auto new_value = make_param("new_value." + idx, new_token_shape);
        auto mask = make_param("mask." + idx, mask_shape);

        std::shared_ptr<ov::Node> key_path;
        std::shared_ptr<ov::Node> value_path;
        std::shared_ptr<ov::Node> key_concat;
        std::shared_ptr<ov::Node> value_concat;
        std::shared_ptr<op::v0::Parameter> broken_key_context;
        std::shared_ptr<op::v0::Parameter> broken_value_context;

        if (is_broken) {
            // Keep the branch intentionally outside SDPA past-kv pattern matching, but shape-valid.
            broken_key_context = make_param("broken_key_context." + idx, context_shape);
            broken_value_context = make_param("broken_value_context." + idx, context_shape);
            key_path = broken_key_context;
            value_path = broken_value_context;
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

        make_result(key_path->output(0), "present." + idx + ".key");
        make_result(value_path->output(0), "present." + idx + ".value");
        make_result(matmul2->output(0), "attn_out." + idx);

        if (!is_broken) {
            expected.push_back(SdpaNodes{qk,
                                         matmul2,
                                         softmax,
                                         add,
                                         past_key,
                                         past_value,
                                         key_concat,
                                         value_concat});
        }
    }

    add_poison_subgraphs(params, results, poison_count);

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
    std::vector<SdpaNodes> expected;
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

        auto past_key = make_param("past_key_values." + idx + ".key", past_shape);
        auto past_value = make_param("past_key_values." + idx + ".value", past_shape);
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

        make_result(key_reshape->output(0), "present." + idx + ".key");
        make_result(value_reshape->output(0), "present." + idx + ".value");
        make_result(matmul2->output(0), "attn_out_wrapped." + idx);

        expected.push_back(SdpaNodes{qk,
                                     matmul2,
                                     softmax,
                                     add,
                                     past_key,
                                     past_value,
                                     key_concat,
                                     value_concat});
    }

    auto model = std::make_shared<Model>(results, params, "sdpa_pattern_nodes_wrapped_concat_model");
    model->validate_nodes_and_infer_types();
    return {model, expected};
}

TEST(SdpaPatternNodesTest, MissingConcatKeepsSearchStableAndReturnsInvalidPattern) {
    const auto built = build_sdpa_model(/*num_sdpa=*/1, /*miss_key_concat=*/true, /*miss_past_key_param=*/false);

    auto pattern = ov::npuw::util::find_sdpa_pattern_nodes(built.model);
    const auto& expected = built.expected.front();

    EXPECT_FALSE(pattern.is_valid());
    EXPECT_EQ(pattern.matmul1_node, expected.matmul1);
    EXPECT_EQ(pattern.matmul2_node, expected.matmul2);
    EXPECT_EQ(pattern.softmax_node, expected.softmax);
    EXPECT_EQ(pattern.add_node, expected.add);
    EXPECT_EQ(pattern.past_key_param_node, expected.past_key_param);
    EXPECT_EQ(pattern.past_value_param_node, expected.past_value_param);
    EXPECT_EQ(pattern.past_key_concat_node, nullptr);
    EXPECT_EQ(pattern.past_value_concat_node, expected.past_value_concat);
}

TEST(SdpaPatternNodesTest, MissingPastKeyNodeReturnsInvalidPattern) {
    const auto built = build_sdpa_model(/*num_sdpa=*/1, /*miss_key_concat=*/false, /*miss_past_key_param=*/true);

    auto pattern = ov::npuw::util::find_sdpa_pattern_nodes(built.model);

    EXPECT_FALSE(pattern.is_valid());
    EXPECT_NE(pattern.past_key_concat_node, nullptr);
    EXPECT_EQ(pattern.past_key_param_node, nullptr);
}

TEST(SdpaPatternNodesTest, SingleCompletedSdpaReturnsExpectedNodeSet) {
    const auto built = build_sdpa_model(/*num_sdpa=*/1, /*miss_key_concat=*/false, /*miss_past_key_param=*/false);

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

    ASSERT_EQ(patterns.size(), expected.size());
    ASSERT_EQ(patterns.size(), 9u);

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

}  // namespace
