// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "openvino/op/util/op_types.hpp"
#include "utils/model.hpp"
#include "matchers/subgraph/subgraph.hpp"
#include "test_models/model_0.hpp"
#include "test_models/model_1.hpp"
#include "test_models/model_2.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;

inline std::pair<std::shared_ptr<ov::Node>, std::set<std::shared_ptr<ov::Node>>>
get_functional_ops(const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::Node> start_node = nullptr;
    // todo: check get_ordered_ops (work diffent compilation by compilation) and remove his code after
    std::set<std::shared_ptr<ov::Node>> nodes;
    std::vector<std::shared_ptr<ov::Node>> nodes_tmp;
    std::vector<std::shared_ptr<ov::Node>> layer;

    for (const auto& res : model->get_results()) {
        for (size_t i = 0; i < res->inputs().size(); ++i) {
            layer.push_back(res->get_input_node_shared_ptr(i));
        }
    }
    while (!layer.empty()) {
        std::vector<std::shared_ptr<ov::Node>> prev_layer;
        nodes_tmp.insert(nodes_tmp.begin(), layer.begin(), layer.end());
        for (const auto& op : layer) {
            for (size_t i = 0; i < op->inputs().size(); ++i)
                prev_layer.push_back(op->get_input_node_shared_ptr(i));
        }
        layer = prev_layer;
    }
    for (const auto& node : nodes_tmp) {
        if (ov::op::util::is_parameter(node) || ov::op::util::is_output(node)) {
            continue;
        }
        if (start_node == nullptr) {
            start_node = node;
        }
        nodes.insert(node);
    }

    // for (const auto& op : model->get_ordered_ops()) {
    //     if (ov::op::util::is_parameter(op) || ov::op::util::is_output(op)) {
    //         continue;
    //     }
    //     if (start_node == nullptr) {
    //         start_node = op;
    //     }
    //     nodes.insert(op);
    // }
    return { start_node, nodes };
}

TEST(ModelUtilsTest, generate_0) {
    Model_0 test;
    std::shared_ptr<ov::Model> test_model = test.get(), recovered_model;
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

TEST(ModelUtilsTest, generate_1) {
    Model_1 test;
    std::shared_ptr<ov::Model> test_model = test.get(), recovered_model;
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

TEST(ModelUtilsTest, generate_2) {
    Model_2 test;
    std::shared_ptr<ov::Model> test_model = test.get(), recovered_model;
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
