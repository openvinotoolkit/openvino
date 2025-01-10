// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "proxy_tests.hpp"

using namespace ov::proxy::tests;

TEST_F(ProxyTests, default_tensor_from_default_dev) {
    const std::string dev_name = "MOCK";
    auto context = core.get_default_context(dev_name);
    auto tensor = context.create_tensor(ov::element::f32, {});
    EXPECT_EQ("MOCK.0", tensor.get_device_name());
    ASSERT_TRUE(tensor.is<PluginRemoteTensor>());
    auto rem_tensor = tensor.as<PluginRemoteTensor>();
    EXPECT_TRUE(rem_tensor.is_default());
    auto model = create_model_with_reshape();

    auto compiled_model = core.compile_model(model, context);
    auto comp_context = compiled_model.get_context();
    auto comp_tensor = comp_context.create_tensor(ov::element::f32, {});
    EXPECT_EQ("MOCK.0", comp_tensor.get_device_name());
    ASSERT_TRUE(comp_tensor.is<PluginRemoteTensor>());
    auto comp_rem_tensor = comp_tensor.as<PluginRemoteTensor>();
    EXPECT_TRUE(comp_rem_tensor.is_default());

    auto infer_request = compiled_model.create_infer_request();
    auto in_tensor = infer_request.get_input_tensor();
    ASSERT_TRUE(in_tensor.is<ov::RemoteTensor>());
    auto in_rem_tensor = in_tensor.as<ov::RemoteTensor>();
    EXPECT_EQ("MOCK.0", in_rem_tensor.get_device_name());
    ASSERT_TRUE(in_rem_tensor.is<PluginRemoteTensor>());
    auto casted_in_rem_tensor = in_rem_tensor.as<PluginRemoteTensor>();
    EXPECT_TRUE(casted_in_rem_tensor.is_default());
}

TEST_F(ProxyTests, default_tensor_from_main_dev) {
    const std::string dev_name = "MOCK.0";
    auto context = core.get_default_context(dev_name);
    auto tensor = context.create_tensor(ov::element::f32, {});
    EXPECT_EQ("MOCK.0", tensor.get_device_name());
    ASSERT_TRUE(tensor.is<PluginRemoteTensor>());
    auto rem_tensor = tensor.as<PluginRemoteTensor>();
    EXPECT_TRUE(rem_tensor.is_default());
    auto model = create_model_with_reshape();

    auto compiled_model = core.compile_model(model, context);
    auto comp_context = compiled_model.get_context();
    auto comp_tensor = comp_context.create_tensor(ov::element::f32, {});
    EXPECT_EQ("MOCK.0", comp_tensor.get_device_name());
    ASSERT_TRUE(comp_tensor.is<PluginRemoteTensor>());
    auto comp_rem_tensor = comp_tensor.as<PluginRemoteTensor>();
    EXPECT_TRUE(comp_rem_tensor.is_default());

    auto infer_request = compiled_model.create_infer_request();
    auto in_tensor = infer_request.get_input_tensor();
    ASSERT_TRUE(in_tensor.is<ov::RemoteTensor>());
    auto in_rem_tensor = in_tensor.as<ov::RemoteTensor>();
    EXPECT_EQ("MOCK.0", in_rem_tensor.get_device_name());
    ASSERT_TRUE(in_rem_tensor.is<PluginRemoteTensor>());
    auto casted_in_rem_tensor = in_rem_tensor.as<PluginRemoteTensor>();
    EXPECT_TRUE(casted_in_rem_tensor.is_default());
}

TEST_F(ProxyTests, default_tensor_from_second_dev) {
    const std::string dev_name = "MOCK.3";
    auto context = core.get_default_context(dev_name);
    auto tensor = context.create_tensor(ov::element::f32, {});
    EXPECT_EQ("MOCK.3", tensor.get_device_name());
    ASSERT_TRUE(tensor.is<PluginRemoteTensor>());
    auto rem_tensor = tensor.as<PluginRemoteTensor>();
    EXPECT_TRUE(rem_tensor.is_default());
    auto model = create_model_with_subtract();

    auto compiled_model = core.compile_model(model, context);
    auto comp_context = compiled_model.get_context();
    auto comp_tensor = comp_context.create_tensor(ov::element::f32, {});
    EXPECT_EQ("MOCK.3", comp_tensor.get_device_name());
    ASSERT_TRUE(comp_tensor.is<PluginRemoteTensor>());
    auto comp_rem_tensor = comp_tensor.as<PluginRemoteTensor>();
    EXPECT_TRUE(comp_rem_tensor.is_default());

    auto infer_request = compiled_model.create_infer_request();
    auto in_tensor = infer_request.get_input_tensor();
    ASSERT_TRUE(in_tensor.is<ov::RemoteTensor>());
    auto in_rem_tensor = in_tensor.as<ov::RemoteTensor>();
    EXPECT_EQ("MOCK.3", in_rem_tensor.get_device_name());
    ASSERT_TRUE(in_rem_tensor.is<PluginRemoteTensor>());
    auto casted_in_rem_tensor = in_rem_tensor.as<PluginRemoteTensor>();
    EXPECT_TRUE(casted_in_rem_tensor.is_default());
}

TEST_F(ProxyTests, custom_tensor_from_default_dev) {
    const std::string dev_name = "MOCK";
    auto context = core.create_context(dev_name, {{"CUSTOM_CTX", true}});
    auto tensor = context.create_tensor(ov::element::f32, {});
    EXPECT_EQ("MOCK.0", tensor.get_device_name());
    ASSERT_TRUE(tensor.is<PluginRemoteTensor>());
    auto rem_tensor = tensor.as<PluginRemoteTensor>();
    EXPECT_FALSE(rem_tensor.is_default());
    auto model = create_model_with_reshape();

    auto compiled_model = core.compile_model(model, context);
    auto comp_context = compiled_model.get_context();
    auto comp_tensor = comp_context.create_tensor(ov::element::f32, {});
    EXPECT_EQ("MOCK.0", comp_tensor.get_device_name());
    ASSERT_TRUE(comp_tensor.is<PluginRemoteTensor>());
    auto comp_rem_tensor = comp_tensor.as<PluginRemoteTensor>();
    EXPECT_FALSE(comp_rem_tensor.is_default());

    auto infer_request = compiled_model.create_infer_request();
    auto in_tensor = infer_request.get_input_tensor();
    ASSERT_TRUE(in_tensor.is<ov::RemoteTensor>());
    auto in_rem_tensor = in_tensor.as<ov::RemoteTensor>();
    EXPECT_EQ("MOCK.0", in_rem_tensor.get_device_name());
    ASSERT_TRUE(in_rem_tensor.is<PluginRemoteTensor>());
    auto casted_in_rem_tensor = in_rem_tensor.as<PluginRemoteTensor>();
    EXPECT_FALSE(casted_in_rem_tensor.is_default());
}

TEST_F(ProxyTests, custom_tensor_from_main_dev) {
    const std::string dev_name = "MOCK.0";
    auto context = core.create_context(dev_name, {{"CUSTOM_CTX", true}});
    auto tensor = context.create_tensor(ov::element::f32, {});
    EXPECT_EQ("MOCK.0", tensor.get_device_name());
    ASSERT_TRUE(tensor.is<PluginRemoteTensor>());
    auto rem_tensor = tensor.as<PluginRemoteTensor>();
    EXPECT_FALSE(rem_tensor.is_default());
    auto model = create_model_with_reshape();

    auto compiled_model = core.compile_model(model, context);
    auto comp_context = compiled_model.get_context();
    auto comp_tensor = comp_context.create_tensor(ov::element::f32, {});
    EXPECT_EQ("MOCK.0", comp_tensor.get_device_name());
    ASSERT_TRUE(comp_tensor.is<PluginRemoteTensor>());
    auto comp_rem_tensor = comp_tensor.as<PluginRemoteTensor>();
    EXPECT_FALSE(comp_rem_tensor.is_default());

    auto infer_request = compiled_model.create_infer_request();
    auto in_tensor = infer_request.get_input_tensor();
    ASSERT_TRUE(in_tensor.is<ov::RemoteTensor>());
    auto in_rem_tensor = in_tensor.as<ov::RemoteTensor>();
    EXPECT_EQ("MOCK.0", in_rem_tensor.get_device_name());
    ASSERT_TRUE(in_rem_tensor.is<PluginRemoteTensor>());
    auto casted_in_rem_tensor = in_rem_tensor.as<PluginRemoteTensor>();
    EXPECT_FALSE(casted_in_rem_tensor.is_default());
}

TEST_F(ProxyTests, custom_tensor_from_second_dev) {
    const std::string dev_name = "MOCK.3";
    auto context = core.create_context(dev_name, {{"CUSTOM_CTX", true}});
    auto tensor = context.create_tensor(ov::element::f32, {});
    EXPECT_EQ("MOCK.3", tensor.get_device_name());
    ASSERT_TRUE(tensor.is<PluginRemoteTensor>());
    auto rem_tensor = tensor.as<PluginRemoteTensor>();
    EXPECT_FALSE(rem_tensor.is_default());
    auto model = create_model_with_subtract();

    auto compiled_model = core.compile_model(model, context);
    auto comp_context = compiled_model.get_context();
    auto comp_tensor = comp_context.create_tensor(ov::element::f32, {});
    EXPECT_EQ("MOCK.3", comp_tensor.get_device_name());
    ASSERT_TRUE(comp_tensor.is<PluginRemoteTensor>());
    auto comp_rem_tensor = comp_tensor.as<PluginRemoteTensor>();
    EXPECT_FALSE(comp_rem_tensor.is_default());

    auto infer_request = compiled_model.create_infer_request();
    auto in_tensor = infer_request.get_input_tensor();
    ASSERT_TRUE(in_tensor.is<ov::RemoteTensor>());
    auto in_rem_tensor = in_tensor.as<ov::RemoteTensor>();
    EXPECT_EQ("MOCK.3", in_rem_tensor.get_device_name());
    ASSERT_TRUE(in_rem_tensor.is<PluginRemoteTensor>());
    auto casted_in_rem_tensor = in_rem_tensor.as<PluginRemoteTensor>();
    EXPECT_FALSE(casted_in_rem_tensor.is_default());
}
