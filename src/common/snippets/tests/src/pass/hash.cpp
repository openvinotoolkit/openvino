// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/hash.hpp"

#include <gtest/gtest.h>

#include "openvino/opsets/opset1.hpp"

namespace ov::test::snippets {
namespace {

std::shared_ptr<ov::Model> make_model(const ov::AnyMap& rt_info) {
    const auto parameter = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1});
    const auto result = std::make_shared<ov::opset1::Result>(parameter);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
    model->get_rt_info() = rt_info;
    return model;
}

uint64_t get_hash(const ov::AnyMap& rt_info) {
    uint64_t hash = 0;
    ov::snippets::pass::Hash{hash}.run_on_model(make_model(rt_info));
    return hash;
}

}  // namespace

TEST(SnippetsHash, ModelRuntimeInfoKeysAndNestedMaps) {
    const ov::AnyMap first{{"key", "value"}};
    const ov::AnyMap same{{"key", "value"}};
    const ov::AnyMap different_key{{"other_key", "value"}};
    const ov::AnyMap nested{{"outer", ov::AnyMap{{"inner", "value"}}}};
    const ov::AnyMap same_nested{{"outer", ov::AnyMap{{"inner", "value"}}}};
    const ov::AnyMap different_outer_key{{"other_outer", ov::AnyMap{{"inner", "value"}}}};
    const ov::AnyMap different_nested_key{{"outer", ov::AnyMap{{"other_inner", "value"}}}};

    EXPECT_EQ(get_hash(first), get_hash(same));
    EXPECT_NE(get_hash(first), get_hash(different_key));
    EXPECT_EQ(get_hash(nested), get_hash(same_nested));
    EXPECT_NE(get_hash(nested), get_hash(different_outer_key));
    EXPECT_NE(get_hash(nested), get_hash(different_nested_key));
}

TEST(SnippetsHash, ModelRuntimeInfoLeafTypes) {
    const std::vector<ov::Any> values{
        std::string{"1"},
        true,
        int64_t{1},
        uint64_t{1},
        1.F,
        1.,
    };

    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(get_hash(ov::AnyMap{{"value", values[i]}}), get_hash(ov::AnyMap{{"value", values[i]}}));
        for (size_t j = i + 1; j < values.size(); ++j) {
            EXPECT_NE(get_hash(ov::AnyMap{{"value", values[i]}}), get_hash(ov::AnyMap{{"value", values[j]}}));
        }
    }
}

}  // namespace ov::test::snippets
