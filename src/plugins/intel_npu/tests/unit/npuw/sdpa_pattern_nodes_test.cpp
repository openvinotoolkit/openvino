// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
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

TEST(SdpaPatternNodesTest, MissingConcatKeepsSearchStableAndReturnsInvalidPattern) {
    const auto built = build_sdpa_model(/*num_sdpa=*/1, /*miss_key_concat=*/true, /*miss_past_key_param=*/false);

    auto pattern = ov::npuw::util::find_sdpa_pattern_nodes(built.model);

    EXPECT_FALSE(pattern.is_valid());
    EXPECT_NE(pattern.matmul1_node, nullptr);
    EXPECT_NE(pattern.past_key_param_node, nullptr);
    EXPECT_EQ(pattern.past_key_concat_node, nullptr);
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

    ASSERT_EQ(patterns.size(), built.expected.size());
    ASSERT_EQ(patterns.size(), 2u);

    for (size_t i = 0; i < patterns.size(); ++i) {
        ASSERT_TRUE(patterns[i].is_valid()) << "Pattern at index " << i << " must be valid";
        expect_nodes_equal(patterns[i], built.expected[i]);
    }
}

}  // namespace
