// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/op_types.hpp"
#include "utils/model.hpp"
#include "utils/model_comparator.hpp"
#include "matchers/subgraph/subgraph.hpp"
#include "test_models/model_0.hpp"
#include "test_models/model_1.hpp"
#include "test_models/model_2.hpp"
#include "base_test.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;

using ModelUtilsTest = SubgraphsDumperBaseTest;

ov::NodeVector
get_functional_ops(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::shared_ptr<ov::Node>> nodes;
    for (const auto& op : model->get_ordered_ops()) {
        nodes.push_back(op);
    }
    return nodes;
}

TEST_F(ModelUtilsTest, generate_0) {
    Model_0 test;
    std::shared_ptr<ov::Model> test_model = test.get(), recovered_model;
    {
        auto func_ops = get_functional_ops(test_model);
        auto model_with_in_info = ov::util::generate_model(func_ops);
        recovered_model = std::get<0>(model_with_in_info);
    }
    {
        ASSERT_TRUE(ov::util::ModelComparator::get()->match(test_model, recovered_model));
    }
}

TEST_F(ModelUtilsTest, generate_1) {
    Model_1 test;
    std::shared_ptr<ov::Model> test_model = test.get(), recovered_model;
    {
        auto func_ops = get_functional_ops(test_model);
        auto model_with_in_info = ov::util::generate_model(func_ops);
        recovered_model = std::get<0>(model_with_in_info);
    }
    {
        ASSERT_TRUE(ov::util::ModelComparator::get()->match(test_model, recovered_model));
    }
}

TEST_F(ModelUtilsTest, generate_2) {
    Model_2 test;
    std::shared_ptr<ov::Model> test_model = test.get(), recovered_model;
    {
        auto func_ops = get_functional_ops(test_model);
        auto model_with_in_info = ov::util::generate_model(func_ops);
        recovered_model = std::get<0>(model_with_in_info);
        auto in_info = std::get<1>(model_with_in_info);
    }
    {
        ASSERT_TRUE(ov::util::ModelComparator::get()->match(test_model, recovered_model));
    }
}

TEST_F(ModelUtilsTest, align_input_info) {
    Model_0 test_model_0, test_model_1;
    auto in_info_0 = ov::util::get_input_info_by_model(test_model_0.get());
    auto in_info_1 = ov::util::get_input_info_by_model(test_model_1.get());
    ASSERT_NE(in_info_0, in_info_1);
    std::unordered_map<std::string, std::string> a;
    OV_ASSERT_NO_THROW(ov::util::align_input_info(test_model_0.get(), test_model_1.get(),
                                               in_info_0, in_info_1, a));
    auto in_info_ref = ov::util::align_input_info(test_model_0.get(), test_model_1.get(),
                                                  in_info_0, in_info_1, a);
    ASSERT_EQ(in_info_1, in_info_ref);
}

TEST_F(ModelUtilsTest, align_input_info_for_subgraphs) {
    Model_0 model_0, model_1;
    auto test_model_0 = model_0.get();
    auto test_model_1 = model_1.get();
    auto in_info_0 = ov::util::get_input_info_by_model(test_model_0);
    auto in_info_1 = ov::util::get_input_info_by_model(test_model_1);
    ASSERT_NE(in_info_0, in_info_1);
    auto matched_ops = ov::util::ModelComparator::get()->get_matched_ops_in_graphs(test_model_0, test_model_1);
    auto params_0 = test_model_0->get_parameters();
    auto params_1 = test_model_1->get_parameters();
    size_t params_cnt = params_0.size();
    for (size_t param_id = 0; param_id < params_cnt; ++param_id) {
        matched_ops.insert({params_0[param_id]->get_friendly_name(),
                            params_1[param_id]->get_friendly_name()});
    }
    // OV_ASSERT_NO_THROW(ov::util::align_input_info(test_model_0, test_model_1,
    //                                            in_info_0, in_info_1,
    //                                            matched_ops));
    auto ref = ov::util::align_input_info(test_model_0, test_model_1,
                                          in_info_0, in_info_1, matched_ops);
    ASSERT_EQ(in_info_1, ref);
}

TEST_F(ModelUtilsTest, get_input_info_by_model) {
    Model_1 model;
    auto test_model = model.get();
    size_t param_idx = 0;
    std::map<std::string, ov::conformance::InputInfo> ref;
    for (auto& param : test_model->get_parameters()) {
        std::string param_name = "parameter_" + std::to_string(param_idx++);
        param->set_friendly_name(param_name);
        ref.insert({param_name, ov::conformance::InputInfo(param->get_default_output().get_partial_shape())});
    }
    auto cur = ov::util::get_input_info_by_model(test_model);
    ASSERT_EQ(cur, ref);
}

TEST_F(ModelUtilsTest, get_subgraph_set_node) {
    Model_1 model;
    std::unordered_set<std::shared_ptr<ov::Node>> out_ops;
    ov::util::get_subgraph_set_node(out_ops, model.get_test_abs_0());
    auto expected = model.get_out_nodes_after_abs_0();
    std::set<std::shared_ptr<ov::Node>> orig(out_ops.begin(), out_ops.end()),
                                        ref(expected.begin(), expected.end());
    ASSERT_EQ(orig, ref);
}
}  // namespace
