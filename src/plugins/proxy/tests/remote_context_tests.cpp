// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "proxy_tests.hpp"

using namespace ov::proxy::tests;

TEST_F(ProxyTests, get_default_context_from_default_dev) {
    const std::string dev_name = "MOCK";
    auto context = core.get_default_context(dev_name);
    EXPECT_EQ("MOCK.0", context.get_device_name());
    ASSERT_TRUE(context.is<PluginRemoteContext>());
    auto rem_context = context.as<PluginRemoteContext>();
    EXPECT_TRUE(rem_context.is_default());
    auto model = create_model_with_reshape();

    auto compiled_model = core.compile_model(model, context);
    auto comp_context = compiled_model.get_context();
    EXPECT_EQ("MOCK.0", comp_context.get_device_name());
    ASSERT_TRUE(comp_context.is<PluginRemoteContext>());
    auto rem_comp_context = comp_context.as<PluginRemoteContext>();
    EXPECT_TRUE(rem_comp_context.is_default());
}

TEST_F(ProxyTests, get_default_context_from_main_dev) {
    const std::string dev_name = "MOCK.0";
    auto context = core.get_default_context(dev_name);
    EXPECT_EQ("MOCK.0", context.get_device_name());
    ASSERT_TRUE(context.is<PluginRemoteContext>());
    auto rem_context = context.as<PluginRemoteContext>();
    EXPECT_TRUE(rem_context.is_default());
    auto model = create_model_with_reshape();

    auto compiled_model = core.compile_model(model, context);
    auto comp_context = compiled_model.get_context();
    EXPECT_EQ("MOCK.0", comp_context.get_device_name());
    ASSERT_TRUE(comp_context.is<PluginRemoteContext>());
    auto rem_comp_context = comp_context.as<PluginRemoteContext>();
    EXPECT_TRUE(rem_comp_context.is_default());
}

TEST_F(ProxyTests, get_default_context_from_splited_dev) {
    const std::string dev_name = "MOCK.1";
    EXPECT_ANY_THROW(auto context = core.get_default_context(dev_name));
}

TEST_F(ProxyTests, get_default_context_from_second_dev) {
    const std::string dev_name = "MOCK.3";
    auto context = core.get_default_context(dev_name);
    EXPECT_EQ("MOCK.3", context.get_device_name());
    ASSERT_TRUE(context.is<PluginRemoteContext>());
    auto rem_context = context.as<PluginRemoteContext>();
    EXPECT_TRUE(rem_context.is_default());
    auto model = create_model_with_subtract();

    auto compiled_model = core.compile_model(model, context);
    auto comp_context = compiled_model.get_context();
    EXPECT_EQ("MOCK.3", comp_context.get_device_name());
    ASSERT_TRUE(comp_context.is<PluginRemoteContext>());
    auto rem_comp_context = comp_context.as<PluginRemoteContext>();
    EXPECT_TRUE(rem_comp_context.is_default());
}

TEST_F(ProxyTests, create_custom_context_from_default_dev) {
    const std::string dev_name = "MOCK";
    auto context = core.create_context(dev_name, {{"CUSTOM_CTX", true}});
    EXPECT_EQ("MOCK.0", context.get_device_name());
    ASSERT_TRUE(context.is<PluginRemoteContext>());
    auto rem_context = context.as<PluginRemoteContext>();
    EXPECT_FALSE(rem_context.is_default());
    auto model = create_model_with_reshape();

    auto compiled_model = core.compile_model(model, context);
    auto comp_context = compiled_model.get_context();
    EXPECT_EQ("MOCK.0", comp_context.get_device_name());
    ASSERT_TRUE(comp_context.is<PluginRemoteContext>());
    auto rem_comp_context = comp_context.as<PluginRemoteContext>();
    EXPECT_FALSE(rem_comp_context.is_default());
}

TEST_F(ProxyTests, create_custom_context_from_main_dev) {
    const std::string dev_name = "MOCK.0";
    auto context = core.create_context(dev_name, {{"CUSTOM_CTX", true}});
    EXPECT_EQ("MOCK.0", context.get_device_name());
    ASSERT_TRUE(context.is<PluginRemoteContext>());
    auto rem_context = context.as<PluginRemoteContext>();
    EXPECT_FALSE(rem_context.is_default());
    auto model = create_model_with_reshape();

    auto compiled_model = core.compile_model(model, context);
    auto comp_context = compiled_model.get_context();
    EXPECT_EQ("MOCK.0", comp_context.get_device_name());
    ASSERT_TRUE(comp_context.is<PluginRemoteContext>());
    auto rem_comp_context = comp_context.as<PluginRemoteContext>();
    EXPECT_FALSE(rem_comp_context.is_default());
}

TEST_F(ProxyTests, create_custom_context_from_splited_dev) {
    const std::string dev_name = "MOCK.1";
    EXPECT_ANY_THROW(auto context = core.create_context(dev_name, {{"CUSTOM_CTX", true}}));
}

TEST_F(ProxyTests, create_custom_context_from_second_dev) {
    const std::string dev_name = "MOCK.3";
    auto context = core.create_context(dev_name, {{"CUSTOM_CTX", true}});
    EXPECT_EQ("MOCK.3", context.get_device_name());
    ASSERT_TRUE(context.is<PluginRemoteContext>());
    auto rem_context = context.as<PluginRemoteContext>();
    EXPECT_FALSE(rem_context.is_default());
    auto model = create_model_with_subtract();

    auto compiled_model = core.compile_model(model, context);
    auto comp_context = compiled_model.get_context();
    EXPECT_EQ("MOCK.3", comp_context.get_device_name());
    ASSERT_TRUE(comp_context.is<PluginRemoteContext>());
    auto rem_comp_context = comp_context.as<PluginRemoteContext>();
    EXPECT_FALSE(rem_comp_context.is_default());
}

TEST_F(ProxyTests, get_context_from_the_model) {
    const std::string dev_name = "MOCK.3";
    auto model = create_model_with_subtract();

    auto compiled_model = core.compile_model(model, dev_name);
    auto comp_context = compiled_model.get_context();
    EXPECT_EQ(dev_name, comp_context.get_device_name());
    EXPECT_NE(core.get_default_context("MOCK").get_device_name(), comp_context.get_device_name());
    ASSERT_TRUE(comp_context.is<PluginRemoteContext>());
    auto rem_comp_context = comp_context.as<PluginRemoteContext>();
    EXPECT_TRUE(rem_comp_context.is_default());
}
