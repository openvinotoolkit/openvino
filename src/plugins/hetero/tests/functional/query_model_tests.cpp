// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "hetero_tests.hpp"

using namespace ov::hetero::tests;

// AVAILABLE_DEVICES {"MOCK0.0", "MOCK0.1", "MOCK0.2", "MOCK1.0", "MOCK1.0"};

TEST_F(HeteroTests, query_model_on_mock0) {
    const std::string dev_name = "MOCK0.1";
    const auto model = create_model_with_subtract_reshape();
    const auto supported_ops = core.query_model(model, "HETERO", {ov::device::priorities(dev_name)});
    std::unordered_set<std::string> names;
    for (const auto& op : model->get_ops()) {
        names.insert(op->get_friendly_name());
    }
    for (const auto& op : supported_ops) {
        EXPECT_EQ(op.second, dev_name);
        names.erase(op.first);
    }
    EXPECT_EQ(1, names.size());
    EXPECT_EQ("sub", *names.begin());
}

TEST_F(HeteroTests, query_model_on_mock1) {
    const std::string dev_name = "MOCK1.1";
    const auto model = create_model_with_subtract_reshape();
    // This WA is needed because mock plugins are loaded one by one
    EXPECT_NO_THROW(core.get_available_devices());
    const auto supported_ops = core.query_model(model, "HETERO", {ov::device::priorities(dev_name)});
    std::unordered_set<std::string> names;
    for (const auto& op : model->get_ops()) {
        names.insert(op->get_friendly_name());
    }
    for (const auto& op : supported_ops) {
        EXPECT_EQ(op.second, dev_name);
        names.erase(op.first);
    }
    const std::vector<std::string> unmarked_names = {"reshape_val", "reshape", "res"};
    EXPECT_EQ(unmarked_names.size(), names.size());
    for (auto& name : unmarked_names) {
        auto it = names.find(name);
        if (it != names.end())
            names.erase(it);
    }
    EXPECT_EQ(0, names.size());
}

TEST_F(HeteroTests, query_model_on_mixed) {
    const std::string dev_name0 = "MOCK0.3";
    const std::string dev_name1 = "MOCK1.2";
    ov::AnyMap config = {ov::device::priorities(dev_name0 + "," + dev_name1)};
    const auto model = create_model_with_subtract_reshape();
    std::set<std::string> supported_ops_mock0;
    for (auto& op : core.query_model(model, dev_name0)) {
        if (op.second == dev_name0)
            supported_ops_mock0.insert(op.first);
    }
    const auto supported_ops = core.query_model(model, "HETERO", config);
    std::unordered_set<std::string> names;
    for (const auto& op : model->get_ops()) {
        names.insert(op->get_friendly_name());
    }
    for (const auto& op : supported_ops) {
        if (supported_ops_mock0.count(op.first))
            EXPECT_EQ(op.second, dev_name0);
        else
            EXPECT_EQ(op.second, dev_name1);
        names.erase(op.first);
    }
    EXPECT_EQ(0, names.size());
}

TEST_F(HeteroTests, query_dynamic_model_on_mixed) {
    const std::string dev_name0 = "MOCK0.3";
    const std::string dev_name1 = "MOCK1.2";
    ov::AnyMap config = {ov::device::priorities(dev_name0 + "," + dev_name1)};
    const auto model = create_model_with_subtract_reshape(true);
    std::set<std::string> supported_ops_mock0;
    for (auto& op : core.query_model(model, dev_name0)) {
        if (op.second == dev_name0)
            supported_ops_mock0.insert(op.first);
    }
    const auto supported_ops = core.query_model(model, "HETERO", config);
    std::unordered_set<std::string> names;
    for (const auto& op : model->get_ops()) {
        names.insert(op->get_friendly_name());
    }
    for (const auto& op : supported_ops) {
        if (supported_ops_mock0.count(op.first))
            EXPECT_EQ(op.second, dev_name0);
        else
            EXPECT_EQ(op.second, dev_name1);
        names.erase(op.first);
    }
    EXPECT_EQ(1, names.size());
    // fallback plugin doesn't support dynamism
    ASSERT_TRUE(names.count("sub"));
}
