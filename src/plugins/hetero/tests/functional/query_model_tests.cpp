// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_test_utils/test_constants.hpp"
#include "hetero_tests.hpp"

namespace ov {
namespace hetero {
namespace tests {

TEST_F(HeteroTests, query_model_on_mock0) {
    const std::string dev_name = "MOCK0.1";
    const auto model = create_model_with_subtract_reshape();
    const auto supported_ops =
        core.query_model(model, ov::test::utils::DEVICE_HETERO, {ov::device::priorities(dev_name)});
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
    const auto supported_ops =
        core.query_model(model, ov::test::utils::DEVICE_HETERO, {ov::device::priorities(dev_name)});
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
    const auto supported_ops = core.query_model(model, ov::test::utils::DEVICE_HETERO, config);
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
    const auto supported_ops = core.query_model(model, ov::test::utils::DEVICE_HETERO, config);
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

TEST_F(HeteroTests, query_model_on_independent_parameter) {
    ov::SupportedOpsMap supported_ops;
    const std::string dev_name = "MOCK0.1";
    const auto model = create_model_with_independent_parameter();
    ASSERT_NO_THROW(supported_ops =
                        core.query_model(model, ov::test::utils::DEVICE_HETERO, {ov::device::priorities(dev_name)}));
    std::unordered_set<std::string> names;
    for (const auto& op : model->get_ops()) {
        names.insert(op->get_friendly_name());
    }
    for (const auto& op : supported_ops) {
        EXPECT_EQ(op.second, dev_name);
        names.erase(op.first);
    }
    EXPECT_EQ(0, names.size());
}

TEST_F(HeteroTests, query_model_by_three_device) {
    const std::string dev_name0 = "MOCKGPU.2";
    const std::string dev_name1 = "MOCKGPU.1";
    const std::string dev_name2 = "MOCKGPU.0";
    std::set<ov::hint::ModelDistributionPolicy> model_policy = {ov::hint::ModelDistributionPolicy::PIPELINE_PARALLEL};
    // This WA is needed because mock plugins are loaded one by one
    EXPECT_NO_THROW(core.get_available_devices());
    const auto model = create_model_with_multi_add();
    const auto supported_ops = core.query_model(model,
                                                ov::test::utils::DEVICE_HETERO,
                                                {ov::device::priorities(dev_name0 + "," + dev_name1 + "," + dev_name2),
                                                 ov::hint::model_distribution_policy(model_policy)});
    std::map<std::string, std::string> expect_result = {{"input", "MOCKGPU.2"},
                                                        {"const_val1", "MOCKGPU.2"},
                                                        {"const_val2", "MOCKGPU.2"},
                                                        {"add1", "MOCKGPU.2"},
                                                        {"add2", "MOCKGPU.2"},
                                                        {"const_val3", "MOCKGPU.1"},
                                                        {"add3", "MOCKGPU.1"},
                                                        {"const_val4", "MOCKGPU.0"},
                                                        {"add4", "MOCKGPU.0"},
                                                        {"res", "MOCKGPU.0"}};
    for (const auto& op : supported_ops) {
        if (expect_result.find(op.first) != expect_result.end()) {
            EXPECT_EQ(op.second, expect_result[op.first]);
        }
    }
}

TEST_F(HeteroTests, query_model_by_two_device) {
    const std::string dev_name0 = "MOCKGPU.2";
    const std::string dev_name1 = "MOCKGPU.0";
    std::set<ov::hint::ModelDistributionPolicy> model_policy = {ov::hint::ModelDistributionPolicy::PIPELINE_PARALLEL};

    // This WA is needed because mock plugins are loaded one by one
    EXPECT_NO_THROW(core.get_available_devices());
    const auto model = create_model_with_multi_add();
    const auto supported_ops = core.query_model(
        model,
        ov::test::utils::DEVICE_HETERO,
        {ov::device::priorities(dev_name0 + "," + dev_name1), ov::hint::model_distribution_policy(model_policy)});
    std::map<std::string, std::string> expect_result = {{"input", "MOCKGPU.2"},
                                                        {"const_val1", "MOCKGPU.2"},
                                                        {"const_val2", "MOCKGPU.2"},
                                                        {"add1", "MOCKGPU.2"},
                                                        {"add2", "MOCKGPU.2"},
                                                        {"const_val3", "MOCKGPU.0"},
                                                        {"add3", "MOCKGPU.0"},
                                                        {"const_val4", "MOCKGPU.0"},
                                                        {"add4", "MOCKGPU.0"},
                                                        {"res", "MOCKGPU.0"}};
    for (const auto& op : supported_ops) {
        if (expect_result.find(op.first) != expect_result.end()) {
            EXPECT_EQ(op.second, expect_result[op.first]);
        }
    }
}
}  // namespace tests
}  // namespace hetero
}  // namespace ov