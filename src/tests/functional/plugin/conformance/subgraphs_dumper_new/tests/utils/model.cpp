// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "openvino/op/util/op_types.hpp"
#include "utils/model.hpp"
#include "test_models.hpp"

#include "matchers/subgraph/subgraph.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;

inline std::pair<std::shared_ptr<ov::Node>, std::set<std::shared_ptr<ov::Node>>>
get_functional_ops(const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::Node> start_node = nullptr;
    std::set<std::shared_ptr<ov::Node>> nodes;
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::op::util::is_parameter(op) || ov::op::util::is_output(op)) {
            continue;
        }
        if (start_node == nullptr) {
            start_node = op;
        }
        nodes.insert(op);
    }
    return { start_node, nodes };
}

TEST(ModelUtilsTest, generate_model_abs_relu_add) {
    std::shared_ptr<ov::Model> test_model = generate_abs_relu_add(), recovered_model;
    {
        std::unordered_set<std::string> checked_ops;
        auto func_ops = get_functional_ops(test_model);
        auto model_with_in_info = generate_model(func_ops.second, func_ops.first, checked_ops);
        recovered_model = model_with_in_info.first;
        for (const auto& op : recovered_model->get_ordered_ops()) {
            if (ov::op::util::is_parameter(op) || ov::op::util::is_constant(op)) {
                ASSERT_TRUE(model_with_in_info.second.count(op->get_friendly_name()));
            }
        }
    }
    {
        SubgraphExtractor extractor;
        ASSERT_TRUE(extractor.match(test_model, recovered_model));
    }
}

TEST(ModelUtilsTest, generate_model_abs_relu_abs_clamp_add) {
    std::shared_ptr<ov::Model> test_model = generate_abs_relu_abs_clamp_add(), recovered_model;
    {
        std::unordered_set<std::string> checked_ops;
        auto func_ops = get_functional_ops(test_model);
        auto model_with_in_info = generate_model(func_ops.second, func_ops.first, checked_ops);
        recovered_model = model_with_in_info.first;
        for (const auto& op : recovered_model->get_ordered_ops()) {
            if (ov::op::util::is_parameter(op) || ov::op::util::is_constant(op)) {
                ASSERT_TRUE(model_with_in_info.second.count(op->get_friendly_name()));
            }
        }
    }
    {
        SubgraphExtractor extractor;
        ASSERT_TRUE(extractor.match(test_model, recovered_model));
    }
}

TEST(ModelUtilsTest, generate_model_abs_clamp_relu_abs_add) {
    std::shared_ptr<ov::Model> test_model = generate_abs_clamp_relu_abs_add(), recovered_model;
    {
        std::unordered_set<std::string> checked_ops;
        auto func_ops = get_functional_ops(test_model);
        auto model_with_in_info = generate_model(func_ops.second, func_ops.first, checked_ops);
        recovered_model = model_with_in_info.first;
        for (const auto& op : recovered_model->get_ordered_ops()) {
            if (ov::op::util::is_parameter(op) || ov::op::util::is_constant(op)) {
                ASSERT_TRUE(model_with_in_info.second.count(op->get_friendly_name()));
            }
        }
    }
    {
        SubgraphExtractor extractor;
        ASSERT_TRUE(extractor.match(test_model, recovered_model));
    }
}

}  // namespace
