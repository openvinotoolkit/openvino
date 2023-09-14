// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_collector.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/graph_comparator.hpp"
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
std::shared_ptr<ov::Model> create_subgraph_add() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::op::v1::Add>(param, const_value);
    add->set_friendly_name("add");
    auto result = std::make_shared<ov::op::v0::Result>(add);
    result->set_friendly_name("add_0_result");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}
std::shared_ptr<ov::Model> create_subgraph_sub() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("add_0_parameter");
    auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto sub = std::make_shared<ov::op::v1::Subtract>(param, const_value);
    sub->set_friendly_name("sub");
    auto result = std::make_shared<ov::op::v0::Result>(sub);
    result->set_friendly_name("sub_0_result");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}
std::shared_ptr<ov::Model> create_subgraph_reshape() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("sub_0_parameter");
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
        m_submodels = {};
        m_submodels.push_back(create_subgraph_add());
        m_submodels.push_back(create_subgraph_sub());
        m_submodels.push_back(create_subgraph_reshape());
    }

    void TearDown() override {}

protected:
    std::shared_ptr<ov::Model> m_model;
    std::vector<std::shared_ptr<ov::Model>> m_submodels;
};

TEST_F(SubgraphCollectorTest, check_subgraphs) {
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
    const std::map<std::string, std::string> expected_parameter_to_prev_result_names = {
        {"sub_0_parameter", "sub_0_result"},
        {"add_0_parameter", "add_0_result"},
    };
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
    SubgraphCollector subgraph_collector(m_model, affinities);
    auto actual_subgraphs_ids = subgraph_collector.get_subgraph_ids();
    for (auto& op : ordered_ops) {
        ASSERT_EQ(actual_subgraphs_ids.at(op), exptected_subgraphs_ids.at(op));
    }
    // Subgraphs are not collected yet
    ASSERT_EQ(0, subgraph_collector.get_subgraph_parameter_to_prev_result().size());
    // Collect subgraphs
    auto actual_subgraphs = subgraph_collector.get_ordered_subgraphs();
    const size_t submodels_number = 3;
    ASSERT_EQ(submodels_number, actual_subgraphs.size());
    auto get_submodel = [&](size_t i) {
        return std::make_shared<ov::Model>(actual_subgraphs.at(i)._results, actual_subgraphs.at(i)._parameters);
    };
    std::pair<bool, std::string> res;
    for (size_t i = 0; i < submodels_number; i++) {
        auto submodel = get_submodel(i);
        auto res = compare_functions(m_submodels.at(i), submodel);
        ASSERT_TRUE(res.first) << res.second;
    }
    auto actual_parameter_to_prev_result = subgraph_collector.get_subgraph_parameter_to_prev_result();
    ASSERT_EQ(actual_parameter_to_prev_result.size(), expected_parameter_to_prev_result_names.size());
    for (auto& item : actual_parameter_to_prev_result) {
        ASSERT_EQ(item.second->get_friendly_name(),
                  expected_parameter_to_prev_result_names.at(item.first->get_friendly_name()));
    }
}
