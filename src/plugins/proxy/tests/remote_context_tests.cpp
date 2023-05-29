
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "proxy_tests.hpp"

using namespace ov::proxy::tests;

TEST_F(ProxyTests, get_default_context_from_default_dev) {
    const std::string dev_name = "MOCK";
    auto context = core.get_default_context(dev_name);
    EXPECT_EQ("MOCK.0", context.get_device_name());
    auto model = create_model_with_reshape();
    ASSERT_NO_THROW(core.compile_model(model, context));
}

TEST_F(ProxyTests, get_default_context_from_main_dev) {
    const std::string dev_name = "MOCK.0";
    auto context = core.get_default_context(dev_name);
    EXPECT_EQ("MOCK.0", context.get_device_name());
    auto model = create_model_with_reshape();
    ASSERT_NO_THROW(core.compile_model(model, context));
}

TEST_F(ProxyTests, get_default_context_from_splited_dev) {
    const std::string dev_name = "MOCK.1";
    EXPECT_ANY_THROW(auto context = core.get_default_context(dev_name));
    // EXPECT_EQ("MOCK.1", context.get_device_name());
}

TEST_F(ProxyTests, get_default_context_from_second_dev) {
    const std::string dev_name = "MOCK.3";
    auto context = core.get_default_context(dev_name);
    EXPECT_EQ("MOCK.3", context.get_device_name());
    auto model = create_model_with_subtract();
    ASSERT_NO_THROW(core.compile_model(model, context));
}
