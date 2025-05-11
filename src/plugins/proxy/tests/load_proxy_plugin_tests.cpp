// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/proxy/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "proxy_tests.hpp"

using namespace ov::proxy::tests;

TEST_F(ProxyTests, alias_for_the_same_name) {
    register_plugin_support_reshape(core,
                                    "CBD",
                                    {{ov::proxy::configuration::alias.name(), "CBD"},
                                     {ov::proxy::configuration::fallback.name(), "DEK"},
                                     {ov::proxy::configuration::priority.name(), 0}});
    register_plugin_support_subtract(core, "DEK", {{ov::proxy::configuration::alias.name(), "CBD"}});
    auto available_devices = core.get_available_devices();
    // 0, 1, 2 is ABC plugin
    // 1, 3, 4 is BDE plugin
    // ABC doesn't support subtract operation
    std::unordered_map<std::string, std::string> mock_reference_dev = {{"CBD.0", "CBD_ov_internal"},
                                                                       {"CBD.1", "CBD_ov_internal DEK"},
                                                                       {"CBD.2", "CBD_ov_internal"}};
    for (const auto& it : mock_reference_dev) {
        EXPECT_EQ(core.get_property(it.first, ov::device::priorities), it.second);
    }
    for (const auto& dev : available_devices) {
        auto it = mock_reference_dev.find(dev);
        if (it != mock_reference_dev.end()) {
            mock_reference_dev.erase(it);
        }
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());
}

TEST_F(ProxyTests, alias_for_the_same_name_with_custom_internal_name_inversed_order) {
    register_plugin_support_subtract(core, "DEK", {{ov::proxy::configuration::alias.name(), "CBD"}});
    register_plugin_support_reshape(core,
                                    "CBD",
                                    {{ov::proxy::configuration::alias.name(), "CBD"},
                                     {ov::proxy::configuration::fallback.name(), "DEK"},
                                     {ov::proxy::configuration::internal_name.name(), "CBD_INTERNAL"},
                                     {ov::proxy::configuration::priority.name(), 0}});
    auto available_devices = core.get_available_devices();
    // 0, 1, 2 is ABC plugin
    // 1, 3, 4 is BDE plugin
    // ABC doesn't support subtract operation
    std::unordered_map<std::string, std::string> mock_reference_dev = {{"CBD.0", "CBD_INTERNAL"},
                                                                       {"CBD.1", "CBD_INTERNAL DEK"},
                                                                       {"CBD.2", "CBD_INTERNAL"}};
    for (const auto& it : mock_reference_dev) {
        EXPECT_EQ(core.get_property(it.first, ov::device::priorities), it.second);
    }
    for (const auto& dev : available_devices) {
        auto it = mock_reference_dev.find(dev);
        if (it != mock_reference_dev.end()) {
            mock_reference_dev.erase(it);
        }
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());
}

TEST_F(ProxyTests, alias_for_the_same_name_with_custom_internal_name) {
    register_plugin_support_reshape(core,
                                    "CBD",
                                    {{ov::proxy::configuration::alias.name(), "CBD"},
                                     {ov::proxy::configuration::fallback.name(), "DEK"},
                                     {ov::proxy::configuration::internal_name.name(), "CBD_INTERNAL"},
                                     {ov::proxy::configuration::priority.name(), 0}});
    register_plugin_support_subtract(core, "DEK", {{ov::proxy::configuration::alias.name(), "CBD"}});
    auto available_devices = core.get_available_devices();
    // 0, 1, 2 is ABC plugin
    // 1, 3, 4 is BDE plugin
    // ABC doesn't support subtract operation
    std::unordered_map<std::string, std::string> mock_reference_dev = {{"CBD.0", "CBD_INTERNAL"},
                                                                       {"CBD.1", "CBD_INTERNAL DEK"},
                                                                       {"CBD.2", "CBD_INTERNAL"}};
    for (const auto& it : mock_reference_dev) {
        EXPECT_EQ(core.get_property(it.first, ov::device::priorities), it.second);
    }
    for (const auto& dev : available_devices) {
        auto it = mock_reference_dev.find(dev);
        if (it != mock_reference_dev.end()) {
            mock_reference_dev.erase(it);
        }
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());
}

TEST_F(ProxyTests, fallback_to_alias_name) {
    register_plugin_support_reshape(
        core,
        "CBD",
        {{ov::proxy::configuration::alias.name(), "CBD"}, {ov::proxy::configuration::priority.name(), 0}});
    register_plugin_support_subtract(core,
                                     "DEK",
                                     {{ov::proxy::configuration::alias.name(), "CBD"},
                                      {ov::proxy::configuration::fallback.name(), "CBD"},
                                      {ov::proxy::configuration::priority.name(), 1}});
    auto available_devices = core.get_available_devices();
    // 0, 1, 2 is ABC plugin
    // 1, 3, 4 is BDE plugin
    // ABC doesn't support subtract operation
    std::unordered_map<std::string, std::string> mock_reference_dev = {{"CBD.0", "CBD_ov_internal"},
                                                                       {"CBD.1", "DEK CBD_ov_internal"},
                                                                       {"CBD.2", "CBD_ov_internal"},
                                                                       {"CBD.3", "DEK"},
                                                                       {"CBD.4", "DEK"}};
    for (const auto& it : mock_reference_dev) {
        EXPECT_EQ(core.get_property(it.first, ov::device::priorities), it.second);
    }
    for (const auto& dev : available_devices) {
        auto it = mock_reference_dev.find(dev);
        if (it != mock_reference_dev.end()) {
            mock_reference_dev.erase(it);
        }
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());
}

TEST_F(ProxyTests, fallback_to_alias_name_with_custom_internal_name) {
    register_plugin_support_reshape(core,
                                    "CBD",
                                    {{ov::proxy::configuration::alias.name(), "CBD"},
                                     {ov::proxy::configuration::internal_name.name(), "CBD_INTERNAL"},
                                     {ov::proxy::configuration::priority.name(), 0}});
    register_plugin_support_subtract(core,
                                     "DEK",
                                     {{ov::proxy::configuration::alias.name(), "CBD"},
                                      {ov::proxy::configuration::fallback.name(), "CBD"},
                                      {ov::proxy::configuration::priority.name(), 1}});
    auto available_devices = core.get_available_devices();
    // 0, 1, 2 is ABC plugin
    // 1, 3, 4 is BDE plugin
    // ABC doesn't support subtract operation
    std::unordered_map<std::string, std::string> mock_reference_dev = {{"CBD.0", "CBD_INTERNAL"},
                                                                       {"CBD.1", "DEK CBD_INTERNAL"},
                                                                       {"CBD.2", "CBD_INTERNAL"},
                                                                       {"CBD.3", "DEK"},
                                                                       {"CBD.4", "DEK"}};
    for (const auto& it : mock_reference_dev) {
        EXPECT_EQ(core.get_property(it.first, ov::device::priorities), it.second);
    }
    for (const auto& dev : available_devices) {
        auto it = mock_reference_dev.find(dev);
        if (it != mock_reference_dev.end()) {
            mock_reference_dev.erase(it);
        }
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());
}

TEST_F(ProxyTests, fallback_to_alias_name_with_custom_internal_name_inverted_order) {
    register_plugin_support_subtract(core,
                                     "DEK",
                                     {{ov::proxy::configuration::alias.name(), "CBD"},
                                      {ov::proxy::configuration::fallback.name(), "CBD"},
                                      {ov::proxy::configuration::priority.name(), 1}});
    register_plugin_support_reshape(core,
                                    "CBD",
                                    {{ov::proxy::configuration::alias.name(), "CBD"},
                                     {ov::proxy::configuration::internal_name.name(), "CBD_INTERNAL"},
                                     {ov::proxy::configuration::priority.name(), 0}});
    auto available_devices = core.get_available_devices();
    // 0, 1, 2 is ABC plugin
    // 1, 3, 4 is BDE plugin
    // ABC doesn't support subtract operation
    std::unordered_map<std::string, std::string> mock_reference_dev = {{"CBD.0", "CBD_INTERNAL"},
                                                                       {"CBD.1", "DEK CBD_INTERNAL"},
                                                                       {"CBD.2", "CBD_INTERNAL"},
                                                                       {"CBD.3", "DEK"},
                                                                       {"CBD.4", "DEK"}};
    for (const auto& it : mock_reference_dev) {
        EXPECT_EQ(core.get_property(it.first, ov::device::priorities), it.second);
    }
    for (const auto& dev : available_devices) {
        auto it = mock_reference_dev.find(dev);
        if (it != mock_reference_dev.end()) {
            mock_reference_dev.erase(it);
        }
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());
}

TEST_F(ProxyTests, load_proxy_on_plugin_without_devices_with_the_same_name) {
    auto available_devices = core.get_available_devices();
    register_plugin_without_devices(
        core,
        "CBD",
        {{ov::proxy::configuration::alias.name(), "CBD"}, {ov::proxy::configuration::priority.name(), 0}});
    available_devices = core.get_available_devices();
    for (const auto& dev : available_devices) {
        EXPECT_NE(dev, "CBD");
    }
    available_devices = core.get_property("CBD", ov::available_devices);
    EXPECT_TRUE(available_devices.empty());
}

TEST_F(ProxyTests, load_proxy_on_plugin_without_devices) {
    auto available_devices = core.get_available_devices();
    register_plugin_without_devices(
        core,
        "Internal_CBD",
        {{ov::proxy::configuration::alias.name(), "CBD"}, {ov::proxy::configuration::priority.name(), 0}});
    available_devices = core.get_available_devices();
    for (const auto& dev : available_devices) {
        EXPECT_NE(dev, "CBD");
    }
    available_devices = core.get_property("CBD", ov::available_devices);
    EXPECT_TRUE(available_devices.empty());
}

TEST_F(ProxyTests, get_available_devices) {
    auto available_devices = core.get_available_devices();
    // 0, 1, 2 is ABC plugin
    // 1, 3, 4 is BDE plugin
    // ABC doesn't support subtract operation
    std::unordered_map<std::string, std::string> mock_reference_dev = {{"MOCK.0", "ABC"},
                                                                       {"MOCK.1", "ABC BDE"},
                                                                       {"MOCK.2", "ABC"},
                                                                       {"MOCK.3", "BDE"},
                                                                       {"MOCK.4", "BDE"}};
    for (const auto& it : mock_reference_dev) {
        EXPECT_EQ(core.get_property(it.first, ov::device::priorities), it.second);
    }
    for (const auto& dev : available_devices) {
        EXPECT_FALSE(dev.find("ABC") != std::string::npos);
        EXPECT_FALSE(dev.find("BDE") != std::string::npos);
        auto it = mock_reference_dev.find(dev);
        if (it != mock_reference_dev.end()) {
            mock_reference_dev.erase(it);
        }
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());
}

TEST_F(ProxyTests, get_available_devices_with_low_level_plugin) {
    ov::AnyMap config;
    config[ov::proxy::alias_for.name()] = "BDE";
    // Change device priority
    core.set_property("MOCK", config);
    auto available_devices = core.get_available_devices();
    // 0, 1, 2 is ABC plugin
    // 1, 3, 4 is BDE plugin
    // ABC doesn't support subtract operation
    {
        // We don't change fallback order for hetero case
        std::unordered_map<std::string, std::string> mock_reference_dev = {{"MOCK.0", "ABC BDE"},
                                                                           {"MOCK.1", "ABC BDE"},
                                                                           {"MOCK.2", "ABC BDE"}};
        for (const auto& it : mock_reference_dev) {
            EXPECT_EQ(core.get_property(it.first, ov::device::priorities), it.second);
        }
    }
    std::set<std::string> mock_reference_dev = {"ABC.ABC_1", "ABC.ABC_2", "ABC.ABC_3", "MOCK.0", "MOCK.1", "MOCK.2"};
    for (const auto& dev : available_devices) {
        if (mock_reference_dev.find(dev) != mock_reference_dev.end()) {
            mock_reference_dev.erase(dev);
        }
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());
}

TEST_F(ProxyTests, load_proxy_without_several_devices) {
    ov::AnyMap config;
    config[ov::proxy::alias_for.name()] = std::vector<std::string>{"Fake1", "Fake2"};
    config[ov::proxy::device_priorities.name()] = std::vector<std::string>{"Fake1:0", "Fake2:1"};
    config[ov::device::priorities.name()] = std::vector<std::string>{"Fake1", "Fake2"};
    // Change device priority
    core.set_property("MOCK", config);
    auto available_devices = core.get_available_devices();
    EXPECT_THROW(core.get_property("MOCK", ov::device::priorities), ov::Exception);
    std::set<std::string> mock_reference_dev = {"MOCK", "MOCK.0", "MOCK.1", "MOCK.2"};
    for (const auto& dev : available_devices) {
        if (mock_reference_dev.find(dev) != mock_reference_dev.end()) {
            mock_reference_dev.erase(dev);
        }
    }
    // Mock devices shouldn't be found
    EXPECT_EQ(mock_reference_dev.size(), 4);
}

TEST_F(ProxyTests, load_proxy_without_devices) {
    ov::AnyMap config;
    config[ov::proxy::alias_for.name()] = "Fake";
    config[ov::proxy::device_priorities.name()] = "Fake:1";
    config[ov::device::priorities.name()] = std::vector<std::string>{"Fake"};
    // Change device priority
    core.set_property("MOCK", config);
    auto available_devices = core.get_available_devices();
    EXPECT_THROW(core.get_property("MOCK", ov::device::priorities), ov::Exception);
    std::set<std::string> mock_reference_dev = {"MOCK", "MOCK.0", "MOCK.1", "MOCK.2"};
    for (const auto& dev : available_devices) {
        if (mock_reference_dev.find(dev) != mock_reference_dev.end()) {
            mock_reference_dev.erase(dev);
        }
    }
    // Mock devices shouldn't be found
    EXPECT_EQ(mock_reference_dev.size(), 4);
}

TEST_F(ProxyTests, load_proxy_with_unavailable_device) {
    ov::AnyMap config;
    config[ov::proxy::alias_for.name()] = std::vector<std::string>{"Fake", "BDE"};
    config[ov::proxy::device_priorities.name()] = std::vector<std::string>{"Fake:1", "BDE:0"};
    config[ov::device::priorities.name()] = std::vector<std::string>{"Fake", "BDE"};
    // Change device priority
    core.set_property("MOCK", config);
    auto available_devices = core.get_available_devices();
    {
        // We don't change fallback order for hetero case
        std::unordered_map<std::string, std::string> mock_reference_dev = {{"MOCK.0", "BDE"},
                                                                           {"MOCK.1", "BDE"},
                                                                           {"MOCK.2", "BDE"}};
        for (const auto& it : mock_reference_dev) {
            std::cout << it.second << std::endl;
            EXPECT_EQ(core.get_property(it.first, ov::device::priorities), it.second);
        }
    }
    std::set<std::string> mock_reference_dev = {"MOCK.0", "MOCK.1", "MOCK.2"};
    for (const auto& dev : available_devices) {
        if (mock_reference_dev.find(dev) != mock_reference_dev.end()) {
            mock_reference_dev.erase(dev);
        }
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());
}

TEST_F(ProxyTests, get_available_devices_with_disabled_plugin) {
    ov::AnyMap config;
    config[ov::device::priorities.name()] = "BDE";
    // Change device priority
    core.set_property("MOCK", config);
    auto available_devices = core.get_available_devices();
    std::unordered_map<std::string, std::string> mock_reference_dev = {{"MOCK.0", "ABC"},
                                                                       {"MOCK.1", "BDE"},
                                                                       {"MOCK.2", "ABC"},
                                                                       {"MOCK.3", "BDE"},
                                                                       {"MOCK.4", "BDE"}};
    for (const auto& it : mock_reference_dev) {
        EXPECT_EQ(core.get_property(it.first, ov::device::priorities), it.second);
    }
    for (const auto& dev : available_devices) {
        auto it = mock_reference_dev.find(dev);
        if (it != mock_reference_dev.end()) {
            mock_reference_dev.erase(it);
        }
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());
}

TEST_F(ProxyTests, load_and_infer_on_device_without_split_on_default_device) {
    // Model has only add (+ 1) op and reshape
    auto model = create_model_with_reshape();
    auto infer_request = core.compile_model(model, "MOCK").create_infer_request();
    auto input_tensor = create_and_fill_tensor(model->input().get_element_type(), model->input().get_shape());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output_tensor = infer_request.get_output_tensor();
    EXPECT_EQ(input_tensor.get_size(), output_tensor.get_size());
    EXPECT_EQ(input_tensor.get_element_type(), output_tensor.get_element_type());
    EXPECT_NE(memcmp(input_tensor.data(), output_tensor.data(), input_tensor.get_byte_size()), 0);
    // Change input tensor
    {
        auto* data = input_tensor.data<int64_t>();
        for (size_t i = 0; i < input_tensor.get_size(); i++)
            data[i] += 1;
    }
    EXPECT_EQ(memcmp(input_tensor.data(), output_tensor.data(), input_tensor.get_byte_size()), 0);
}

TEST_F(ProxyTests, load_and_infer_on_device_without_split) {
    auto model = create_model_with_subtract();
    auto infer_request = core.compile_model(model, "MOCK.3").create_infer_request();
    auto input_tensor = create_and_fill_tensor(model->input().get_element_type(), model->input().get_shape());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output_tensor = infer_request.get_output_tensor();
    EXPECT_EQ(input_tensor.get_shape(), output_tensor.get_shape());
    EXPECT_EQ(input_tensor.get_element_type(), output_tensor.get_element_type());
    EXPECT_EQ(memcmp(input_tensor.data(), output_tensor.data(), input_tensor.get_byte_size()), 0);
}

TEST_F(ProxyTests, load_on_unsupported_plugin) {
    auto model = create_model_with_subtract();
    EXPECT_EQ(core.get_property("MOCK.0", ov::device::priorities), "ABC");
    EXPECT_THROW(core.compile_model(model, "MOCK.0"), ov::Exception);
}

TEST_F(ProxyTests, load_on_supported_plugin) {
    auto model = create_model_with_subtract();
    EXPECT_EQ(core.get_property("MOCK.3", ov::device::priorities), "BDE");
    EXPECT_NO_THROW(core.compile_model(model, "MOCK.3"));
}

#ifdef HETERO_ENABLED
TEST_F(ProxyTests, load_on_shared_plugin) {
    auto model = create_model_with_subtract();
    EXPECT_EQ(core.get_property("MOCK.1", ov::device::priorities), "ABC BDE");
    EXPECT_NO_THROW(core.compile_model(model, "MOCK.1"));
}

TEST_F(ProxyTests, load_and_infer_on_support_with_hetero_plugin) {
    auto model = create_model_with_subtract();
    auto infer_request = core.compile_model(model, "MOCK.1").create_infer_request();
    auto input_tensor = create_and_fill_tensor(model->input().get_element_type(), model->input().get_shape());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output_tensor = infer_request.get_output_tensor();
    EXPECT_EQ(input_tensor.get_shape(), output_tensor.get_shape());
    EXPECT_EQ(input_tensor.get_element_type(), output_tensor.get_element_type());
    EXPECT_EQ(memcmp(input_tensor.data(), output_tensor.data(), input_tensor.get_byte_size()), 0);
}
#endif
