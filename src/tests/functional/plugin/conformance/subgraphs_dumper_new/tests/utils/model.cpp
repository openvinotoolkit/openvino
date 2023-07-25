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

std::pair<std::shared_ptr<ov::Node>, std::set<std::shared_ptr<ov::Node>>>
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

TEST(ModelUtilsTest, generate_0) {
    Model_0 test;
    std::shared_ptr<ov::Model> test_model = test.get(), recovered_model;
    {
        std::unordered_set<std::string> checked_ops;
        auto func_ops = get_functional_ops(test_model);
        auto model_with_in_info = generate_model(func_ops.second, func_ops.first, checked_ops);
        recovered_model = model_with_in_info.first;
        for (const auto& in : model_with_in_info.second) {
            std::cout << "IN_INFO: " << in.first << std::endl;
        }
        for (const auto& op : recovered_model->get_ordered_ops()) {
            if (ov::op::util::is_parameter(op) || ov::op::util::is_constant(op)) {
                std::cout << "RECOVERED_MODEL: " << op->get_friendly_name() << std::endl;
                // ASSERT_TRUE(model_with_in_info.second.count(op->get_friendly_name()));
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
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::Serialize>("model_1.xml", "model_1.bin");
            manager.run_passes(recovered_model);
            recovered_model->validate_nodes_and_infer_types();
        for (const auto& in : model_with_in_info.second) {
            std::cout << "IN_INFO: " << in.first << std::endl;
        }
        for (const auto& op : recovered_model->get_ordered_ops()) {
            if (ov::op::util::is_parameter(op) || ov::op::util::is_constant(op)) {
                std::cout << "RECOVERED_MODEL: " << op->get_friendly_name() << std::endl;
                // ASSERT_TRUE(model_with_in_info.second.count(op->get_friendly_name()));
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
        for (const auto& in : model_with_in_info.second) {
            std::cout << "IN_INFO: " << in.first << std::endl;
        }
        for (const auto& op : recovered_model->get_ordered_ops()) {
            if (ov::op::util::is_parameter(op) || ov::op::util::is_constant(op)) {
                std::cout << "RECOVERED_MODEL: " << op->get_friendly_name() << std::endl;
                // ASSERT_TRUE(model_with_in_info.second.count(op->get_friendly_name()));
            }
        }
    }
    {
        SubgraphExtractor extractor;
        ASSERT_TRUE(extractor.match(test_model, recovered_model));
    }
}

}  // namespace
