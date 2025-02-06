// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <gtest/gtest.h>

#include <memory>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/iplugin.hpp"

namespace ov {
namespace proxy {
namespace tests {

class PluginRemoteTensor : public ov::RemoteTensor {
public:
    /**
     * @brief Checks that type defined runtime parameters are presented in remote object
     * @param tensor a tensor to check
     */
    static void type_check(const Tensor& tensor) {
        RemoteTensor::type_check(tensor, {{"IS_DEFAULT", {}}});
    }

    bool is_default() {
        return get_params().at("IS_DEFAULT").as<bool>();
    }
};

class PluginRemoteContext : public ov::RemoteContext {
public:
    // Needed to make create_tensor overloads from base class visible for user
    using RemoteContext::create_host_tensor;
    using RemoteContext::create_tensor;
    /**
     * @brief Checks that type defined runtime parameters are presented in remote object
     * @param remote_context A remote context to check
     */
    static void type_check(const RemoteContext& remote_context) {
        RemoteContext::type_check(remote_context, {{"IS_DEFAULT", {}}});
    }

    bool is_default() {
        return get_params().at("IS_DEFAULT").as<bool>();
    }
};

// <ie>
//     <plugins>
//         <plugin name="ABC" location="libmock_abc_plugin.so">
//             <properties>
//                 <property key="DEVICE_ID_PROPERTY" value="DEVICE_UUID"/> // The same by default
//                 <property key="PRIMARY_DEVICE" value="YES"/> // NO by default
//             </properties>
//         </plugin>
//         <plugin name="BDE" location="libmock_bde_plugin.so">
//         </plugin>
//     </plugins>
// </ie>
class ProxyTests : public ::testing::Test {
public:
    ov::Core core;

    void SetUp() override;
    void TearDown() override;

    std::shared_ptr<ov::Model> create_model_with_subtract();
    std::shared_ptr<ov::Model> create_model_with_subtract_reshape();
    std::shared_ptr<ov::Model> create_model_with_subtract_reshape_relu();
    std::shared_ptr<ov::Model> create_model_with_reshape();
    std::shared_ptr<ov::Model> create_model_with_add();
    ov::Tensor create_and_fill_tensor(const ov::element::Type& type, const ov::Shape& shape);

protected:
    void register_plugin_without_devices(ov::Core& core, const std::string& device_name, const ov::AnyMap& properties);
    void register_plugin_support_reshape(ov::Core& core, const std::string& device_name, const ov::AnyMap& properties);
    void register_plugin_support_subtract(ov::Core& core, const std::string& device_name, const ov::AnyMap& properties);

private:
    template <class T>
    ov::Tensor create_tensor(const ov::element::Type& type, const ov::Shape& shape) {
        ov::Tensor tensor(type, shape);
        T* data = tensor.data<T>();
        for (size_t i = 0; i < tensor.get_size(); i++) {
            data[i] = static_cast<T>(i);
        }
        return tensor;
    }
    std::vector<std::shared_ptr<ov::IPlugin>> m_mock_plugins;
    std::shared_ptr<void> m_so;

    void reg_plugin(ov::Core& core,
                    std::shared_ptr<ov::IPlugin>& plugin,
                    const std::string& device_name,
                    const ov::AnyMap& properties);
};

}  // namespace tests
}  // namespace proxy
}  // namespace ov
