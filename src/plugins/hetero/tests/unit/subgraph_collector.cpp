// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_collector.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "op/device_subgraph.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/ops.hpp"

using namespace ov::hetero;

namespace {
std::shared_ptr<ov::Model> create_test_model() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::op::v1::Add>(param, const_value);
    add->set_friendly_name("add");
    auto subtract = std::make_shared<ov::op::v1::Subtract>(add, const_value);
    subtract->set_friendly_name("sub");
    auto reshape_val = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    reshape_val->set_friendly_name("reshape_val");
    auto reshape = std::make_shared<ov::op::v1::Reshape>(subtract, reshape_val, true);
    reshape->set_friendly_name("reshape");
    auto result = std::make_shared<ov::op::v0::Result>(reshape);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}
std::shared_ptr<ov::Model> create_test_model2() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add1 = std::make_shared<ov::op::v1::Add>(param, const_value);
    add1->set_friendly_name("add1");
    auto add2 = std::make_shared<ov::op::v1::Add>(add1, const_value);
    add2->set_friendly_name("add2");
    auto subtract = std::make_shared<ov::op::v1::Subtract>(add2, const_value);
    subtract->set_friendly_name("sub");
    auto add3 = std::make_shared<ov::op::v1::Add>(add1, subtract);
    add3->set_friendly_name("add3");
    auto reshape_val = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    reshape_val->set_friendly_name("reshape_val");
    auto reshape = std::make_shared<ov::op::v1::Reshape>(add3, reshape_val, true);
    reshape->set_friendly_name("reshape");
    auto result = std::make_shared<ov::op::v0::Result>(reshape);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}
std::shared_ptr<ov::Model> create_subgraph_add() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::op::v1::Add>(param, const_value);
    add->set_friendly_name("add");
    return std::make_shared<ov::Model>(ov::OutputVector{add->output(0)}, ov::ParameterVector{param});
}
std::shared_ptr<ov::Model> create_subgraph_add_add() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add1 = std::make_shared<ov::op::v1::Add>(param, const_value);
    add1->set_friendly_name("add1");
    auto add2 = std::make_shared<ov::op::v1::Add>(add1, const_value);
    add2->set_friendly_name("add2");
    return std::make_shared<ov::Model>(ov::OutputVector{add2->output(0), add1->output(0)}, ov::ParameterVector{param});
}
std::shared_ptr<ov::Model> create_subgraph_sub() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto sub = std::make_shared<ov::op::v1::Subtract>(param, const_value);
    sub->set_friendly_name("sub");
    return std::make_shared<ov::Model>(ov::OutputVector{sub->output(0)}, ov::ParameterVector{param});
}
std::shared_ptr<ov::Model> create_subgraph_add_reshape() {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    auto add = std::make_shared<ov::op::v1::Add>(param0, param1);
    add->set_friendly_name("add_second");
    auto reshape_val = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    reshape_val->set_friendly_name("reshape_val");
    auto reshape = std::make_shared<ov::op::v1::Reshape>(add, reshape_val, true);
    reshape->set_friendly_name("reshape");
    auto result = std::make_shared<ov::op::v0::Result>(reshape);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param0, param1});
}
std::shared_ptr<ov::Model> create_subgraph_reshape() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    auto reshape_val = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    reshape_val->set_friendly_name("reshape_val");
    auto reshape = std::make_shared<ov::op::v1::Reshape>(param, reshape_val, true);
    reshape->set_friendly_name("reshape");
    auto result = std::make_shared<ov::op::v0::Result>(reshape);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}
}  // namespace

class SubgraphCollectorTest : public testing::Test {
    void SetUp() override {
        m_model = create_test_model();
        m_model_ref = m_model->clone();
        m_submodels = {};
        m_submodels.push_back(create_subgraph_add());
        m_submodels.push_back(create_subgraph_sub());
        m_submodels.push_back(create_subgraph_reshape());
    }

    void TearDown() override {}

protected:
    std::shared_ptr<ov::Model> m_model;
    std::shared_ptr<ov::Model> m_model_ref;
    std::vector<std::shared_ptr<ov::Model>> m_submodels;
};

TEST_F(SubgraphCollectorTest, submodel_split_and_merge) {
    const std::map<std::string, std::string> supported_ops = {
        {"input", "MOCK.0"},
        {"const_val", "MOCK.0"},
        {"add", "MOCK.0"},
        {"sub", "MOCK.1"},
        {"reshape_val", "MOCK.0"},
        {"reshape", "MOCK.0"},
        {"res", "MOCK.0"},
    };
    const std::map<std::string, SubgraphCollector::SubgraphId> expected_ids = {
        {"input", 0},
        {"const_val", 0},
        {"add", 0},
        {"sub", 2},
        {"reshape_val", 3},
        {"reshape", 3},
        {"res", 3},
    };
    const SubgraphsMappingInfo exptected_mapping_info = {{// inputs to submodel inputs
                                                          NodeInfo{0, 0}},
                                                         {// outputs to submodel outpts
                                                          NodeInfo{2, 0}},
                                                         {// submodel input to previous output
                                                          {NodeInfo{1, 0}, NodeInfo{0, 0}},
                                                          {NodeInfo{2, 0}, NodeInfo{1, 0}}}};
    SubgraphCollector::AffinitiesMap affinities;
    SubgraphCollector::SubgraphIdsMap exptected_subgraphs_ids;
    const auto ordered_ops = m_model->get_ordered_ops();
    for (const auto& node : ordered_ops) {
        const auto name = node->get_friendly_name();
        OPENVINO_ASSERT(supported_ops.count(name));
        affinities[node] = supported_ops.at(name);
        OPENVINO_ASSERT(expected_ids.count(name));
        exptected_subgraphs_ids[node] = expected_ids.at(name);
    }
    // Collect subgraphs
    SubgraphCollector subgraph_collector(m_model, affinities);
    auto actual_subgraphs_ids = subgraph_collector.get_subgraph_ids();
    ASSERT_EQ(exptected_subgraphs_ids, actual_subgraphs_ids);
    ov::hetero::SubgraphsVector actual_subgraphs;
    ov::hetero::SubgraphsMappingInfo actual_mapping_info;
    std::tie(actual_subgraphs, actual_mapping_info) = subgraph_collector.run();

    ASSERT_EQ(3, actual_subgraphs.size());
    std::vector<std::shared_ptr<ov::Model>> actual_submodels;
    for (auto& actual_subgraph : actual_subgraphs) {
        actual_submodels.push_back(std::make_shared<ov::Model>(actual_subgraph._results, actual_subgraph._parameters));
    }
    for (size_t i = 0; i < actual_submodels.size(); i++) {
        auto res = compare_functions(m_submodels.at(i), actual_submodels.at(i));
        ASSERT_TRUE(res.first) << res.second;
    }

    // Check mapping info
    ASSERT_EQ(exptected_mapping_info._inputs_to_submodels_inputs, actual_mapping_info._inputs_to_submodels_inputs);
    ASSERT_EQ(exptected_mapping_info._outputs_to_submodels_outputs, actual_mapping_info._outputs_to_submodels_outputs);
    ASSERT_EQ(exptected_mapping_info._submodels_input_to_prev_output,
              actual_mapping_info._submodels_input_to_prev_output);

    // Merge submodels into one model back
    ov::hetero::merge_submodels(actual_submodels, actual_mapping_info._submodels_input_to_prev_output);
    ASSERT_EQ(1, actual_submodels.size());
    auto& actual_model = actual_submodels[0];
    auto res = compare_functions(m_model_ref, actual_model);
    ASSERT_TRUE(res.first) << res.second;
}

TEST_F(SubgraphCollectorTest, submodel_replacement_first_device) {
    const std::map<std::string, std::string> supported_ops_mock0 = {
        {"input", "MOCK.0"},
        {"const_val", "MOCK.0"},
        {"add", "MOCK.0"},
        {"reshape_val", "MOCK.0"},
        {"reshape", "MOCK.0"},
        {"res", "MOCK.0"},
    };
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
        param->set_friendly_name("input");

        auto device_subgraph0 = std::make_shared<ov::hetero::op::DeviceSubgraph>(ov::OutputVector{param->output(0)},
                                                                                 m_submodels.at(0),
                                                                                 "MOCK.0");
        device_subgraph0->set_friendly_name("device_subgraph0");

        auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
        const_value->set_friendly_name("const_val");
        auto subtract = std::make_shared<ov::op::v1::Subtract>(device_subgraph0->output(0), const_value);
        subtract->set_friendly_name("sub");

        auto device_subgraph1 = std::make_shared<ov::hetero::op::DeviceSubgraph>(ov::OutputVector{subtract->output(0)},
                                                                                 m_submodels.at(2),
                                                                                 "MOCK.0");
        device_subgraph1->set_friendly_name("device_subgraph1");

        auto result = std::make_shared<ov::op::v0::Result>(device_subgraph1->output(0));
        result->set_friendly_name("res");
        m_model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
    auto supported_ops = supported_ops_mock0;
    auto actual_mapping_info = ov::hetero::mask_model_subgraphs_by_ops(m_model, supported_ops, false, "TEST");
    auto res = compare_functions(m_model_ref, m_model);
    ASSERT_TRUE(res.first) << res.second;

    // Only first device was replace, mapping is unavailable
    ASSERT_TRUE(actual_mapping_info.empty());
}

TEST_F(SubgraphCollectorTest, submodel_replacement_both_devices) {
    const std::map<std::string, std::string> supported_ops_mock0 = {
        {"input", "MOCK.0"},
        {"const_val", "MOCK.0"},
        {"add", "MOCK.0"},
        {"reshape_val", "MOCK.0"},
        {"reshape", "MOCK.0"},
        {"res", "MOCK.0"},
    };
    const std::map<std::string, std::string> supported_ops_mock1 = {
        {"sub", "MOCK.1"},
    };
    const SubgraphsMappingInfo exptected_mapping_info = {{// inputs to submodel inputs
                                                          NodeInfo{0, 0}},
                                                         {// outputs to submodel outpts
                                                          NodeInfo{2, 0}},
                                                         {// submodel input to previous output
                                                          {NodeInfo{1, 0}, NodeInfo{0, 0}},
                                                          {NodeInfo{2, 0}, NodeInfo{1, 0}}}};
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
        param->set_friendly_name("input");

        auto device_subgraph0 = std::make_shared<ov::hetero::op::DeviceSubgraph>(ov::OutputVector{param->output(0)},
                                                                                 m_submodels.at(0),
                                                                                 "MOCK.0");
        device_subgraph0->set_friendly_name("device_subgraph0");

        auto device_subgraph1 =
            std::make_shared<ov::hetero::op::DeviceSubgraph>(ov::OutputVector{device_subgraph0->output(0)},
                                                             m_submodels.at(1),
                                                             "MOCK.1");
        device_subgraph1->set_friendly_name("device_subgraph1");

        auto device_subgraph2 =
            std::make_shared<ov::hetero::op::DeviceSubgraph>(ov::OutputVector{device_subgraph1->output(0)},
                                                             m_submodels.at(2),
                                                             "MOCK.0");
        device_subgraph2->set_friendly_name("device_subgraph2");

        auto result = std::make_shared<ov::op::v0::Result>(device_subgraph2->output(0));
        result->set_friendly_name("res");
        m_model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
    ov::hetero::SubgraphsMappingInfo actual_mapping_info;
    auto supported_ops = supported_ops_mock0;
    ASSERT_EQ(6, supported_ops.size());
    actual_mapping_info = ov::hetero::mask_model_subgraphs_by_ops(m_model, supported_ops, false, "TEST");
    // + 2 subgraphs + 2 params + 2 results
    ASSERT_EQ(12, supported_ops.size());
    // Adds mock1 supported operations
    supported_ops.insert(supported_ops_mock1.begin(), supported_ops_mock1.end());
    ASSERT_EQ(13, supported_ops.size());
    actual_mapping_info = ov::hetero::mask_model_subgraphs_by_ops(m_model, supported_ops, false);
    // + 1 subgraphs + 1 param + 1 result
    ASSERT_EQ(16, supported_ops.size());
    auto res = compare_functions(m_model_ref, m_model);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_EQ(exptected_mapping_info._inputs_to_submodels_inputs, actual_mapping_info._inputs_to_submodels_inputs);
    ASSERT_EQ(exptected_mapping_info._outputs_to_submodels_outputs, actual_mapping_info._outputs_to_submodels_outputs);
    ASSERT_EQ(exptected_mapping_info._submodels_input_to_prev_output,
              actual_mapping_info._submodels_input_to_prev_output);
}

class SubgraphCollectorTest2 : public SubgraphCollectorTest {
    void SetUp() override {
        m_model = create_test_model2();
        m_model_ref = m_model->clone();
        m_submodels = {};
        m_submodels.push_back(create_subgraph_add_add());
        m_submodels.push_back(create_subgraph_sub());
        m_submodels.push_back(create_subgraph_add_reshape());
    }
};

TEST_F(SubgraphCollectorTest2, submodel_split_and_merge) {
    const std::map<std::string, std::string> supported_ops = {
        {"input", "MOCK.0"},
        {"const_val", "MOCK.0"},
        {"add1", "MOCK.0"},
        {"add2", "MOCK.0"},
        {"sub", "MOCK.1"},
        {"add3", "MOCK.0"},
        {"reshape_val", "MOCK.0"},
        {"reshape", "MOCK.0"},
        {"res", "MOCK.0"},
    };
    const SubgraphsMappingInfo exptected_mapping_info = {{// inputs to submodel inputs
                                                          NodeInfo{0, 0}},
                                                         {// outputs to submodel outpts
                                                          NodeInfo{2, 0}},
                                                         {// submodel input to previous output
                                                          {NodeInfo{1, 0}, NodeInfo{0, 0}},
                                                          {NodeInfo{2, 0}, NodeInfo{0, 1}},
                                                          {NodeInfo{2, 1}, NodeInfo{1, 0}}}};
    SubgraphCollector::AffinitiesMap affinities;
    const auto ordered_ops = m_model->get_ordered_ops();
    for (const auto& node : ordered_ops) {
        const auto name = node->get_friendly_name();
        OPENVINO_ASSERT(supported_ops.count(name));
        affinities[node] = supported_ops.at(name);
    }
    // Collect subgraphs
    SubgraphCollector subgraph_collector(m_model, affinities);
    ov::hetero::SubgraphsVector actual_subgraphs;
    ov::hetero::SubgraphsMappingInfo actual_mapping_info;
    std::tie(actual_subgraphs, actual_mapping_info) = subgraph_collector.run();

    ASSERT_EQ(3, actual_subgraphs.size());
    std::vector<std::shared_ptr<ov::Model>> actual_submodels;
    for (auto& actual_subgraph : actual_subgraphs) {
        actual_submodels.push_back(std::make_shared<ov::Model>(actual_subgraph._results, actual_subgraph._parameters));
    }
    for (size_t i = 0; i < actual_submodels.size(); i++) {
        auto res = compare_functions(m_submodels.at(i), actual_submodels.at(i));
        ASSERT_TRUE(res.first) << res.second;
    }

    // Check mapping info
    ASSERT_EQ(exptected_mapping_info._inputs_to_submodels_inputs, actual_mapping_info._inputs_to_submodels_inputs);
    ASSERT_EQ(exptected_mapping_info._outputs_to_submodels_outputs, actual_mapping_info._outputs_to_submodels_outputs);
    ASSERT_EQ(exptected_mapping_info._submodels_input_to_prev_output,
              actual_mapping_info._submodels_input_to_prev_output);

    // Merge submodels into one model back
    ov::hetero::merge_submodels(actual_submodels, actual_mapping_info._submodels_input_to_prev_output);
    ASSERT_EQ(1, actual_submodels.size());
    auto& actual_model = actual_submodels[0];
    auto res = compare_functions(m_model_ref, actual_model);
    ASSERT_TRUE(res.first) << res.second;
}

TEST_F(SubgraphCollectorTest2, submodel_replacement_first_device) {
    const std::map<std::string, std::string> supported_ops_mock0 = {
        {"input", "MOCK.0"},
        {"const_val", "MOCK.0"},
        {"add1", "MOCK.0"},
        {"add2", "MOCK.0"},
        {"add3", "MOCK.0"},
        {"reshape_val", "MOCK.0"},
        {"reshape", "MOCK.0"},
        {"res", "MOCK.0"},
    };
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
        param->set_friendly_name("input");

        auto device_subgraph0 = std::make_shared<ov::hetero::op::DeviceSubgraph>(ov::OutputVector{param->output(0)},
                                                                                 m_submodels.at(0),
                                                                                 "MOCK.0");
        device_subgraph0->set_friendly_name("device_subgraph0");

        auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
        const_value->set_friendly_name("const_val");
        auto subtract = std::make_shared<ov::op::v1::Subtract>(device_subgraph0->output(0), const_value);
        subtract->set_friendly_name("sub");

        auto device_subgraph2 = std::make_shared<ov::hetero::op::DeviceSubgraph>(
            ov::OutputVector{device_subgraph0->output(0), subtract->output(0)},
            m_submodels.at(2),
            "MOCK.0");
        device_subgraph2->set_friendly_name("device_subgraph2");

        auto result = std::make_shared<ov::op::v0::Result>(device_subgraph2->output(0));
        result->set_friendly_name("res");
        m_model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
    auto supported_ops = supported_ops_mock0;
    ASSERT_EQ(8, supported_ops.size());
    auto actual_mapping_info = ov::hetero::mask_model_subgraphs_by_ops(m_model, supported_ops, false, "TEST");
    // + 2 subgraphs + 3 params + 3 results
    ASSERT_EQ(16, supported_ops.size());
    auto res = compare_functions(m_model_ref, m_model);
    ASSERT_TRUE(res.first) << res.second;

    // Only first device was replace, mapping is unavailable
    ASSERT_TRUE(actual_mapping_info.empty());
}

TEST_F(SubgraphCollectorTest2, submodel_replacement_both_devices) {
    const std::map<std::string, std::string> supported_ops_mock0 = {
        {"input", "MOCK.0"},
        {"const_val", "MOCK.0"},
        {"add1", "MOCK.0"},
        {"add2", "MOCK.0"},
        {"add3", "MOCK.0"},
        {"reshape_val", "MOCK.0"},
        {"reshape", "MOCK.0"},
        {"res", "MOCK.0"},
    };
    const std::map<std::string, std::string> supported_ops_mock1 = {
        {"sub", "MOCK.1"},
    };
    const SubgraphsMappingInfo exptected_mapping_info = {{// inputs to submodel inputs
                                                          NodeInfo{0, 0}},
                                                         {// outputs to submodel outpts
                                                          NodeInfo{2, 0}},
                                                         {// submodel input to previous output
                                                          {NodeInfo{1, 0}, NodeInfo{0, 0}},
                                                          {NodeInfo{2, 0}, NodeInfo{0, 1}},
                                                          {NodeInfo{2, 1}, NodeInfo{1, 0}}}};
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
        param->set_friendly_name("input");

        auto device_subgraph0 = std::make_shared<ov::hetero::op::DeviceSubgraph>(ov::OutputVector{param->output(0)},
                                                                                 m_submodels.at(0),
                                                                                 "MOCK.0");
        device_subgraph0->set_friendly_name("device_subgraph0");

        auto device_subgraph1 =
            std::make_shared<ov::hetero::op::DeviceSubgraph>(ov::OutputVector{device_subgraph0->output(1)},
                                                             m_submodels.at(1),
                                                             "MOCK.1");
        device_subgraph1->set_friendly_name("device_subgraph1");

        auto device_subgraph2 = std::make_shared<ov::hetero::op::DeviceSubgraph>(
            ov::OutputVector{device_subgraph0->output(0), device_subgraph1->output(0)},
            m_submodels.at(2),
            "MOCK.0");
        device_subgraph2->set_friendly_name("device_subgraph2");

        auto result = std::make_shared<ov::op::v0::Result>(device_subgraph2->output(0));
        result->set_friendly_name("res");
        m_model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
    ov::hetero::SubgraphsMappingInfo actual_mapping_info;
    auto supported_ops = supported_ops_mock0;
    ASSERT_EQ(8, supported_ops.size());
    actual_mapping_info = ov::hetero::mask_model_subgraphs_by_ops(m_model, supported_ops, false, "TEST");
    // + 2 subgraphs + 3 params + 3 results
    ASSERT_EQ(16, supported_ops.size());
    // Adds mock1 supported operations
    supported_ops.insert(supported_ops_mock1.begin(), supported_ops_mock1.end());
    ASSERT_EQ(17, supported_ops.size());
    actual_mapping_info = ov::hetero::mask_model_subgraphs_by_ops(m_model, supported_ops, false);
    // + 1 subgraphs + 1 param + 1 result
    ASSERT_EQ(20, supported_ops.size());
    auto res = compare_functions(m_model_ref, m_model);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_EQ(exptected_mapping_info._inputs_to_submodels_inputs, actual_mapping_info._inputs_to_submodels_inputs);
    ASSERT_EQ(exptected_mapping_info._outputs_to_submodels_outputs, actual_mapping_info._outputs_to_submodels_outputs);
    ASSERT_EQ(exptected_mapping_info._submodels_input_to_prev_output,
              actual_mapping_info._submodels_input_to_prev_output);
}

TEST_F(SubgraphCollectorTest, submodel_with_different_affinity_parameter) {
    const std::map<std::string, std::string> supported_ops_with_affinity = {
        {"input", "MOCK.0"},
        {"const_val", "MOCK.0"},
        {"add", "MOCK.1"},
        {"reshape_val", "MOCK.0"},
        {"reshape", "MOCK.0"},
        {"res", "MOCK.0"},
    };
    auto supported_ops = supported_ops_with_affinity;
    OV_ASSERT_NO_THROW(ov::hetero::mask_model_subgraphs_by_ops(m_model, supported_ops, false, "TEST"));
}

TEST_F(SubgraphCollectorTest, submodel_with_constant_subgraphs) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{1, 3, 16, 16});
    input->set_friendly_name("input");
    auto convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);
    convert->set_friendly_name("convert");

    auto constant1 = ov::op::v0::Constant::create(ov::element::f32, {}, {2.f});
    constant1->set_friendly_name("constant1");

    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, constant1);
    mul->set_friendly_name("mul");

    auto constant2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 3, 2});
    constant2->set_friendly_name("constant2");
    auto transpose = std::make_shared<ov::op::v1::Transpose>(mul, constant2);
    transpose->set_friendly_name("transpose");
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(transpose);
    shapeOf->set_friendly_name("shapeOf");

    auto reshape_val = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    reshape_val->set_friendly_name("reshape_val");
    auto reshape = std::make_shared<ov::op::v1::Reshape>(shapeOf, reshape_val, true);
    reshape->set_friendly_name("reshape");

    auto zero = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    zero->set_friendly_name("zero");
    auto one = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    one->set_friendly_name("one");
    auto gather = std::make_shared<ov::op::v8::Gather>(reshape, one, zero);
    gather->set_friendly_name("gather");

    auto result = std::make_shared<ov::op::v0::Result>(gather);
    result->set_friendly_name("result");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    const std::map<std::string, std::string> supported_ops_with_affinity = {
        {"input", "MOCK.0"},
        {"convert", "MOCK.0"},
        {"mul", "MOCK.0"},
        {"constant1", "MOCK.0"},
        {"constant2", "MOCK.0"},
        {"transpose", "MOCK.1"},
        {"shapeOf", "MOCK.0"},
        {"reshape_val", "MOCK.0"},
        {"reshape", "MOCK.1"},
        {"zero", "MOCK.0"},
        {"one", "MOCK.0"},
        {"gather", "MOCK.1"},
        {"result", "MOCK.0"},
    };

    auto supported_ops = supported_ops_with_affinity;
    ov::hetero::SubgraphsVector ordered_subgraphs;
    ov::hetero::SubgraphsMappingInfo actual_mapping_info;
    std::tie(ordered_subgraphs, actual_mapping_info) = get_model_subgraphs(model, supported_ops, true, false);
    for (const auto& subgraph : ordered_subgraphs) {
        std::set<std::string> node_set;
        auto sub_model = std::make_shared<ov::Model>(subgraph._results, subgraph._sinks, subgraph._parameters);
        for (auto& node : sub_model->get_ordered_ops()) {
            node_set.insert(node->get_friendly_name());
        }
        ASSERT_EQ(node_set.count("transpose"), node_set.count("constant2"));
        ASSERT_EQ(node_set.count("reshape"), node_set.count("reshape_val"));
        ASSERT_EQ(node_set.count("gather"), node_set.count("zero"));
        ASSERT_EQ(node_set.count("gather"), node_set.count("one"));
        if (node_set.count("transpose") || node_set.count("reshape") || node_set.count("gather")) {
            ASSERT_EQ(subgraph._affinity, "MOCK.1");
        }
    }
}

TEST_F(SubgraphCollectorTest, merge_independent_submodel) {
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
    param1->set_friendly_name("input1");
    auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
    param2->set_friendly_name("input2");
    auto const_value1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {1});
    const_value1->set_friendly_name("const_val1");
    auto add1 = std::make_shared<ov::op::v1::Add>(param1, const_value1);
    add1->set_friendly_name("add1");
    auto const_value2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {1});
    const_value2->set_friendly_name("const_val1");
    auto add2 = std::make_shared<ov::op::v1::Add>(add1, const_value2);
    add2->set_friendly_name("add2");
    auto result = std::make_shared<ov::op::v0::Result>(add2);
    result->set_friendly_name("res");
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param1, param2});

    const std::map<std::string, std::string> supported_ops = {
        {"input1", "MOCK.0"},
        {"input2", "MOCK.0"},
        {"const_val1", "MOCK.0"},
        {"add1", "MOCK.0"},
        {"const_val2", "MOCK.1"},
        {"add2", "MOCK.1"},
        {"res", "MOCK.1"},
    };
    SubgraphCollector::AffinitiesMap affinities;
    const auto ordered_ops = model->get_ordered_ops();
    for (const auto& node : ordered_ops) {
        const auto name = node->get_friendly_name();
        OPENVINO_ASSERT(supported_ops.count(name));
        affinities[node] = supported_ops.at(name);
    }
    // Collect subgraphs
    SubgraphCollector subgraph_collector(model, affinities);
    ov::hetero::SubgraphsVector actual_subgraphs;
    ov::hetero::SubgraphsMappingInfo actual_mapping_info;
    std::tie(actual_subgraphs, actual_mapping_info) = subgraph_collector.run();

    ASSERT_EQ(3, actual_subgraphs.size());
    std::vector<std::shared_ptr<ov::Model>> actual_submodels;
    for (auto& actual_subgraph : actual_subgraphs) {
        actual_submodels.push_back(std::make_shared<ov::Model>(actual_subgraph._results, actual_subgraph._parameters));
    }
    // Merge submodels into one model back
    OV_ASSERT_NO_THROW(
        ov::hetero::merge_submodels(actual_submodels, actual_mapping_info._submodels_input_to_prev_output));
    ASSERT_EQ(1, actual_submodels.size());
}