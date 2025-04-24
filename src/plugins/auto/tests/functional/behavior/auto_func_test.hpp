// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <thread>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/auto/properties.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/iplugin.hpp"

namespace ov {
namespace auto_plugin {
namespace tests {

#define ASSERT_THROW_WITH_MESSAGE(code, expected_exception, expected_message)       \
    do {                                                                            \
        try {                                                                       \
            { code; }                                                               \
            FAIL() << "no exception occured" << std::endl;                          \
        } catch (const expected_exception& e) {                                     \
            EXPECT_THAT(e.what(), testing::HasSubstr(expected_message));            \
        } catch (const std::exception& e) {                                         \
            FAIL() << "an unexpected exception occured: " << e.what() << std::endl; \
        } catch (...) {                                                             \
            FAIL() << "an unknown exception occured" << std::endl;                  \
        }                                                                           \
    } while (0);

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

class AutoFuncTests : public ::testing::Test {
public:
    ov::Core core;

    void SetUp() override;
    void TearDown() override;

    ov::Tensor create_and_fill_tensor(const ov::element::Type& type, const ov::Shape& shape);

protected:
    void register_plugin_mock_cpu(ov::Core& core, const std::string& device_name, const ov::AnyMap& properties);
    void register_plugin_mock_cpu_compile_slower(ov::Core& core,
                                                 const std::string& device_name,
                                                 const ov::AnyMap& properties);
    void register_plugin_mock_gpu(ov::Core& core, const std::string& device_name, const ov::AnyMap& properties);
    void register_plugin_mock_gpu_compile_slower(ov::Core& core,
                                                 const std::string& device_name,
                                                 const ov::AnyMap& properties);
    std::shared_ptr<ov::Model> model_can_batch;
    std::shared_ptr<ov::Model> model_cannot_batch;
    std::string cache_path;

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
    std::shared_ptr<ov::Model> create_model_with_batch_possible();
    std::shared_ptr<ov::Model> create_model_with_reshape();
};

class ThreadingTest {
public:
    static void runParallel(std::function<void(void)> func,
                            const unsigned int iterations = 100,
                            const unsigned int threadsNum = 8) {
        std::vector<std::thread> threads(threadsNum);

        for (auto& thread : threads) {
            thread = std::thread([&]() {
                for (unsigned int i = 0; i < iterations; ++i) {
                    func();
                }
            });
        }

        for (auto& thread : threads) {
            if (thread.joinable())
                thread.join();
        }
    }
};
}  // namespace tests
}  // namespace auto_plugin
}  // namespace ov