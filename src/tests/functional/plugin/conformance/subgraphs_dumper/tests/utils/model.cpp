// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/op_types.hpp"
#include "utils/model.hpp"
#include "matchers/subgraph/subgraph.hpp"
#include "test_models/model_0.hpp"
#include "test_models/model_1.hpp"
#include "test_models/model_2.hpp"
#include "base_test.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;

using ModelUtilsTest = SubgraphsDumperBaseTest;

std::set<std::shared_ptr<ov::Node>>
get_functional_ops(const std::shared_ptr<ov::Model>& model) {
    std::set<std::shared_ptr<ov::Node>> nodes;
    for (const auto& op : model->get_ordered_ops()) {
        nodes.insert(op);
    }
    return nodes;
}

TEST_F(ModelUtilsTest, generate_0) {
    Model_0 test;
    std::shared_ptr<ov::Model> test_model = test.get(), recovered_model;
    {
        std::unordered_set<std::string> checked_ops;
        auto func_ops = get_functional_ops(test_model);
        auto model_with_in_info = generate_model(func_ops, checked_ops, "test_extractor");
        recovered_model = std::get<0>(model_with_in_info);
    }
    {
        SubgraphExtractor extractor;
        if (!extractor.match(test_model, recovered_model)) {
            ov::pass::Manager manager;
            {
                manager.register_pass<ov::pass::Serialize>("/Users/iefode/repo/temp/model_0.xml", "/Users/iefode/repo/temp/model_0.xml");
                manager.run_passes(recovered_model);
                recovered_model->validate_nodes_and_infer_types();
            }

            {
                manager.register_pass<ov::pass::Serialize>("/Users/iefode/repo/temp/orig_model_0.xml", "/Users/iefode/repo/temp/orig_model_0.xml");
                manager.run_passes(test_model);
                test_model->validate_nodes_and_infer_types();
            }
        }
        ASSERT_TRUE(extractor.match(test_model, recovered_model));
    }
}

TEST_F(ModelUtilsTest, generate_1) {
    Model_1 test;
    std::shared_ptr<ov::Model> test_model = test.get(), recovered_model;
    {
        std::unordered_set<std::string> checked_ops;
        auto func_ops = get_functional_ops(test_model);
        auto model_with_in_info = generate_model(func_ops, checked_ops, "test_extractor");
        recovered_model = std::get<0>(model_with_in_info);
    }
    {
        SubgraphExtractor extractor;
        if (!extractor.match(test_model, recovered_model)) {
            ov::pass::Manager manager;
            {
                manager.register_pass<ov::pass::Serialize>("/Users/iefode/repo/temp/model_1.xml", "/Users/iefode/repo/temp/model_1.xml");
                manager.run_passes(recovered_model);
                recovered_model->validate_nodes_and_infer_types();
            }
            {
                manager.register_pass<ov::pass::Serialize>("/Users/iefode/repo/temp/orig_model_1.xml", "/Users/iefode/repo/temp/orig_model_1.xml");
                manager.run_passes(test_model);
                test_model->validate_nodes_and_infer_types();
            }
        }
        ASSERT_TRUE(extractor.match(test_model, recovered_model));
    }
}

TEST_F(ModelUtilsTest, generate_2) {
    Model_2 test;
    std::shared_ptr<ov::Model> test_model = test.get(), recovered_model;
    {
        std::unordered_set<std::string> checked_ops;
        auto func_ops = get_functional_ops(test_model);
        auto model_with_in_info = generate_model(func_ops, checked_ops, "extract_model");
        recovered_model = std::get<0>(model_with_in_info);
        auto in_info = std::get<1>(model_with_in_info);
    }
    {
        SubgraphExtractor extractor;
        if (!extractor.match(test_model, recovered_model)) {
            ov::pass::Manager manager;
            {
                manager.register_pass<ov::pass::Serialize>("/Users/iefode/repo/temp/model_2.xml", "/Users/iefode/repo/temp/model_2.xml");
                manager.run_passes(recovered_model);
                recovered_model->validate_nodes_and_infer_types();
            }

            {
                manager.register_pass<ov::pass::Serialize>("/Users/iefode/repo/temp/orig_model_2.xml", "/Users/iefode/repo/temp/orig_model_2.xml");
                manager.run_passes(test_model);
                test_model->validate_nodes_and_infer_types();
            }
        }
        ASSERT_TRUE(extractor.match(test_model, recovered_model));
    }
}

}  // namespace
