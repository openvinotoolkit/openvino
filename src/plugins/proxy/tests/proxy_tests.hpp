// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <gtest/gtest.h>

#include <memory>

#include "openvino/runtime/core.hpp"

namespace ov {
namespace proxy {
namespace tests {

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

    std::shared_ptr<ov::Model> create_model_with_subtract();
    std::shared_ptr<ov::Model> create_model_with_subtract_reshape();
    std::shared_ptr<ov::Model> create_model_with_subtract_reshape_relu();
    std::shared_ptr<ov::Model> create_model_with_reshape();
    ov::Tensor create_and_fill_tensor(const ov::element::Type& type, const ov::Shape& shape);

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
    void register_plugin_support_reshape(ov::Core& core, const std::string& device_name, const ov::AnyMap& properties);
    void register_plugin_support_subtract(ov::Core& core, const std::string& device_name, const ov::AnyMap& properties);
};

}  // namespace tests
}  // namespace proxy
}  // namespace ov
