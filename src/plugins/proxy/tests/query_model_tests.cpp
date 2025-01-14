// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "proxy_tests.hpp"

using namespace ov::proxy::tests;

// AVAILABLE_DEVICES {"MOCK.0", "MOCK.1", "MOCK.2", "MOCK.3", "MOCK.4"};
// 1 is shared device
TEST_F(ProxyTests, query_model_on_abc) {
    const std::string dev_name = "MOCK.0";
    const auto model = create_model_with_subtract_reshape();
    auto supported_ops = core.query_model(model, dev_name);
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

TEST_F(ProxyTests, query_model_on_bde) {
    const std::string dev_name = "MOCK.4";
    const auto model = create_model_with_subtract_reshape();
    auto supported_ops = core.query_model(model, dev_name);
    std::unordered_set<std::string> names;
    for (const auto& op : model->get_ops()) {
        names.insert(op->get_friendly_name());
    }
    for (const auto& op : supported_ops) {
        EXPECT_EQ(op.second, dev_name);
        names.erase(op.first);
    }
    EXPECT_EQ(1, names.size());
    EXPECT_EQ("reshape", *names.begin());
}

#ifdef HETERO_ENABLED
TEST_F(ProxyTests, query_model_on_mixed) {
    const std::string dev_name = "MOCK.1";
    const auto model = create_model_with_subtract_reshape();
    auto supported_ops = core.query_model(model, dev_name);
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
#endif
