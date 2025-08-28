// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "common_test_utils/matcher.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_tools.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/pass/pattern/multi_matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::pass;
using namespace ov::pass::pattern;
using namespace ov::pass::pattern::op;

namespace ov::test {

// Match multiple independent MatMul nodes
TEST(MultiMatcherTest, matches_multiple_matmuls) {
    auto input1 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4});
    auto input2 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4});
    auto input3 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4});

    auto weight1 = ov::op::v0::Constant::create(element::f32, Shape{4, 4}, {0});
    auto weight2 = ov::op::v0::Constant::create(element::f32, Shape{4, 4}, {0});
    auto weight3 = ov::op::v0::Constant::create(element::f32, Shape{4, 4}, {0});

    auto mm1 = std::make_shared<ov::op::v0::MatMul>(input1, weight1);
    auto mm2 = std::make_shared<ov::op::v0::MatMul>(input2, weight2);
    auto mm3 = std::make_shared<ov::op::v0::MatMul>(input3, weight3);

    auto model = std::make_shared<Model>(OutputVector{mm1, mm2, mm3}, ParameterVector{input1, input2, input3});

    auto a = any_input();
    auto b = any_input();
    auto pat = wrap_type<ov::op::v0::MatMul>({a, b});

    MultiMatcher matcher("matmul");
    int count = 0;
    matcher.register_patterns({pat}, [&](const auto& matches) {
        count = static_cast<int>(matches.at(pat).size());
    });

    ASSERT_TRUE(matcher.run_on_model(model));
    EXPECT_EQ(count, 3);
}

// Overlapping is allowed, probably will be disabled in the future
TEST(MultiMatcherTest, matches_overlapped_add_and_matmul) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4});
    auto w1 = ov::op::v0::Constant::create(element::f32, Shape{4, 4}, {0});
    auto w2 = ov::op::v0::Constant::create(element::f32, Shape{1, 4}, {1});
    auto mm = std::make_shared<ov::op::v0::MatMul>(input, w1);
    auto add = std::make_shared<ov::op::v1::Add>(mm, w2);

    auto model = std::make_shared<Model>(OutputVector{add}, ParameterVector{input});

    auto a = any_input();
    auto b = any_input();
    auto pat_matmul = wrap_type<ov::op::v0::MatMul>({a, b});
    auto pat_add = wrap_type<ov::op::v1::Add>({pat_matmul, any_input()});

    int adds = 0, matmuls = 0;

    MultiMatcher matcher("mixed");
    matcher.register_patterns({pat_add, pat_matmul}, [&](const auto& matches) {
        if (matches.count(pat_add))
            adds = static_cast<int>(matches.at(pat_add).size());
        if (matches.count(pat_matmul))
            matmuls = static_cast<int>(matches.at(pat_matmul).size());
    });

    ASSERT_TRUE(matcher.run_on_model(model));
    EXPECT_EQ(adds, 1);
    EXPECT_EQ(matmuls, 1);
}

// Overlapping is allowed, probably will be disabled in the future
TEST(MultiMatcherTest, overlap_between_patterns) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4});
    auto w1 = ov::op::v0::Constant::create(element::f32, Shape{4, 4}, {1});
    auto w2 = ov::op::v0::Constant::create(element::f32, Shape{4, 4}, {1});
    auto mm1 = std::make_shared<ov::op::v0::MatMul>(input, w1);
    auto mm2 = std::make_shared<ov::op::v0::MatMul>(input, w2);

    auto model = std::make_shared<Model>(OutputVector{mm1, mm2}, ParameterVector{input});

    auto a = any_input();
    auto b = any_input();
    auto pat = wrap_type<ov::op::v0::MatMul>({a, b});

    int count = 0;
    MultiMatcher matcher("conflict_test");
    matcher.register_patterns({pat}, [&](const auto& matches) {
        count = static_cast<int>(matches.at(pat).size());
    });

    ASSERT_TRUE(matcher.run_on_model(model));
    EXPECT_EQ(count, 2);
}

// No match should be found
TEST(MultiMatcherTest, no_match_found) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto relu = std::make_shared<ov::op::v0::Relu>(input);
    auto model = std::make_shared<Model>(OutputVector{relu}, ParameterVector{input});

    auto a = any_input();
    auto b = any_input();
    auto pat = wrap_type<ov::op::v0::MatMul>({a, b});

    bool callback_invoked = false;
    MultiMatcher matcher("no_match");
    matcher.register_patterns({pat}, [&](const auto&) {
        callback_invoked = true;
    });

    ASSERT_FALSE(matcher.run_on_model(model));
    EXPECT_FALSE(callback_invoked);
}

// Match different operator types (Relu and Sigmoid)
TEST(MultiMatcherTest, matches_multiple_pattern_types) {
    auto input1 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto relu = std::make_shared<ov::op::v0::Relu>(input1);

    auto input2 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(input2);

    auto model = std::make_shared<Model>(OutputVector{relu, sigmoid}, ParameterVector{input1, input2});

    auto relu_pat = wrap_type<ov::op::v0::Relu>({any_input()});
    auto sigm_pat = wrap_type<ov::op::v0::Sigmoid>({any_input()});

    int relu_count = 0;
    int sigm_count = 0;

    MultiMatcher matcher("types");
    matcher.register_patterns({relu_pat, sigm_pat}, [&](const auto& matches) {
        if (matches.count(relu_pat))
            relu_count = static_cast<int>(matches.at(relu_pat).size());
        if (matches.count(sigm_pat))
            sigm_count = static_cast<int>(matches.at(sigm_pat).size());
    });

    ASSERT_TRUE(matcher.run_on_model(model));
    EXPECT_EQ(relu_count, 1);
    EXPECT_EQ(sigm_count, 1);
}

// Multiple matches from one pattern
TEST(MultiMatcherTest, grouped_matches_from_single_pattern) {
    auto input1 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto relu1 = std::make_shared<ov::op::v0::Relu>(input1);

    auto input2 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto relu2 = std::make_shared<ov::op::v0::Relu>(input2);

    auto model = std::make_shared<Model>(OutputVector{relu1, relu2}, ParameterVector{input1, input2});

    auto pat = wrap_type<ov::op::v0::Relu>({any_input()});

    int count = 0;
    MultiMatcher matcher("multi_out");
    matcher.register_patterns({pat}, [&](const auto& matches) {
        count = static_cast<int>(matches.at(pat).size());
    });

    ASSERT_TRUE(matcher.run_on_model(model));
    EXPECT_EQ(count, 2);
}

TEST(MultiMatcherTest, uses_pattern_value_map_contents) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4});
    auto weights = ov::op::v0::Constant::create(element::f32, Shape{4, 4}, {1});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights);

    auto model = std::make_shared<Model>(OutputVector{matmul}, ParameterVector{input});

    auto pat_input = any_input();
    auto pat_weight = any_input();
    auto pat_matmul = wrap_type<ov::op::v0::MatMul>({pat_input, pat_weight});

    bool callback_triggered = false;

    MultiMatcher matcher("value_map");
    matcher.register_patterns({pat_matmul}, [&](const auto& matches) {
        callback_triggered = true;
        ASSERT_EQ(matches.at(pat_matmul).size(), 1);

        const auto& map = matches.at(pat_matmul)[0];
        ASSERT_TRUE(map.count(pat_input));
        ASSERT_TRUE(map.count(pat_weight));
        ASSERT_TRUE(map.count(pat_matmul));

        auto captured_input = map.at(pat_input).get_node_shared_ptr();
        auto captured_weight = map.at(pat_weight).get_node_shared_ptr();
        auto captured_mm = map.at(pat_matmul).get_node_shared_ptr();

        EXPECT_EQ(captured_input, input);
        EXPECT_EQ(captured_weight, weights);
        EXPECT_EQ(captured_mm, matmul);
    });

    ASSERT_TRUE(matcher.run_on_model(model));
    EXPECT_TRUE(callback_triggered);
}

}  // namespace ov::test
