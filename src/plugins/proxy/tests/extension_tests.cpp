// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/relu.hpp"
#include "proxy_tests.hpp"

using namespace ov::proxy::tests;

namespace {
std::unordered_set<std::string> get_unsupported_ops(const ov::Core& core,
                                                    const std::shared_ptr<const ov::Model>& model,
                                                    const std::string& dev_name) {
    auto supported_ops = core.query_model(model, dev_name);
    std::unordered_set<std::string> names;
    for (const auto& op : model->get_ops()) {
        names.insert(op->get_friendly_name());
    }
    for (const auto op : supported_ops) {
        EXPECT_EQ(op.second, dev_name);
        names.erase(op.first);
    }
    return names;
}
}  // namespace

// AVAILABLE_DEVICES {"MOCK.0", "MOCK.1", "MOCK.2", "MOCK.3", "MOCK.4"};
// 1 is shared device
TEST_F(ProxyTests, add_extension_abc) {
    const std::string dev_name = "MOCK.0";

    const auto model = create_model_with_subtract_reshape_relu();
    {
        std::unordered_set<std::string> names = get_unsupported_ops(core, model, dev_name);
        EXPECT_EQ(2, names.size());
        EXPECT_NE(names.end(), names.find("sub"));
        EXPECT_NE(names.end(), names.find("relu"));
    }
    // Add relu to mock plugins as an extension
    {
        core.add_extension<ov::op::v0::Relu>();
        std::unordered_set<std::string> names = get_unsupported_ops(core, model, dev_name);
        EXPECT_EQ(1, names.size());
        EXPECT_NE(names.end(), names.find("sub"));
    }
}

TEST_F(ProxyTests, add_extension_bde) {
    const std::string dev_name = "MOCK.3";

    const auto model = create_model_with_subtract_reshape_relu();
    {
        std::unordered_set<std::string> names = get_unsupported_ops(core, model, dev_name);
        EXPECT_EQ(2, names.size());
        EXPECT_NE(names.end(), names.find("reshape"));
        EXPECT_NE(names.end(), names.find("relu"));
    }
    // Add relu to mock plugins as an extension
    {
        core.add_extension<ov::op::v0::Relu>();
        std::unordered_set<std::string> names = get_unsupported_ops(core, model, dev_name);
        EXPECT_EQ(1, names.size());
        EXPECT_NE(names.end(), names.find("reshape"));
    }
}

TEST_F(ProxyTests, add_extension_mixed) {
    const std::string dev_name = "MOCK.1";

    const auto model = create_model_with_subtract_reshape_relu();
    {
        std::unordered_set<std::string> names = get_unsupported_ops(core, model, dev_name);
        EXPECT_EQ(1, names.size());
        EXPECT_NE(names.end(), names.find("relu"));
    }
    // Add relu to mock plugins as an extension
    {
        core.add_extension<ov::op::v0::Relu>();
        std::unordered_set<std::string> names = get_unsupported_ops(core, model, dev_name);
        EXPECT_EQ(0, names.size());
    }
}
