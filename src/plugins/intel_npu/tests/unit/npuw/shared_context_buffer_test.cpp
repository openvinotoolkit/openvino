// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_context_buffer.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace stubs {

class RemoteTensor : public ov::IRemoteTensor {
public:
    explicit RemoteTensor(std::string device_name) : m_device_name(std::move(device_name)) {}

    const ov::AnyMap& get_properties() const override {
        return m_properties;
    }

    const std::string& get_device_name() const override {
        return m_device_name;
    }

    void set_shape(ov::Shape) override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    const ov::element::Type& get_element_type() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    const ov::Shape& get_shape() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    const ov::Strides& get_strides() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

private:
    std::string m_device_name;
    ov::AnyMap m_properties;
};

class RemoteContext : public ov::IRemoteContext {
public:
    explicit RemoteContext(std::string device_name) : m_device_name(std::move(device_name)) {}

    const std::string& get_device_name() const override {
        return m_device_name;
    }

    const ov::AnyMap& get_property() const override {
        return m_properties;
    }

    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type&,
                                               const ov::Shape&,
                                               const ov::AnyMap&) override {
        return {std::make_shared<RemoteTensor>(m_device_name), nullptr};
    }

private:
    std::string m_device_name;
    ov::AnyMap m_properties;
};

}  // namespace stubs

namespace {

std::vector<ov::SoPtr<ov::IRemoteContext>> get_required_contexts(const std::vector<std::string>& required_devices) {
    std::vector<ov::SoPtr<ov::IRemoteContext>> contexts;
    contexts.reserve(required_devices.size());
    for (const auto& device_prefix : required_devices) {
        contexts.push_back({std::make_shared<stubs::RemoteContext>(device_prefix), nullptr});
    }
    return contexts;
}

TEST(SharedContextBufferTests, ProvidesRemoteTensorsForGpuAndNpuContexts) {
    const size_t byte_size = 4096;
    const size_t alignment = 4096;

    const std::vector<std::string> required_devices = {"GPU", "NPU"};
    auto contexts = get_required_contexts(required_devices);
    if (contexts.size() != required_devices.size()) {
        GTEST_SKIP() << "Required contexts are unavailable. Requested: " << required_devices.size()
                     << ", created: " << contexts.size();
    }

    const auto& gpu_ctx = contexts[0];
    const auto& npu_ctx = contexts[1];
    std::vector<ov::SoPtr<ov::IRemoteContext>> remote_contexts = {gpu_ctx, npu_ctx};

    auto buffer = std::make_shared<ov::npuw::SharedContextBuffer>(byte_size, std::move(remote_contexts), alignment);
    ASSERT_NE(buffer, nullptr);

    auto gpu_tensor = buffer->get_remote_tensors_if_exist(gpu_ctx);
    auto npu_tensor = buffer->get_remote_tensors_if_exist(npu_ctx);

    ASSERT_NE(gpu_tensor, nullptr);
    ASSERT_NE(npu_tensor, nullptr);
    EXPECT_EQ(gpu_tensor->get_device_name().rfind("GPU", 0), 0u);
    EXPECT_EQ(npu_tensor->get_device_name().rfind("NPU", 0), 0u);
}

TEST(SharedContextBufferTests, ProvidesUsableDescriptorWithRemoteContexts) {
    const size_t byte_size = 8192;
    const size_t alignment = 4096;

    const std::vector<std::string> required_devices = {"GPU", "NPU"};
    auto contexts = get_required_contexts(required_devices);
    if (contexts.size() != required_devices.size()) {
        GTEST_SKIP() << "Required contexts are unavailable. Requested: " << required_devices.size()
                     << ", created: " << contexts.size();
    }

    const auto& gpu_ctx = contexts[0];
    const auto& npu_ctx = contexts[1];
    std::vector<ov::SoPtr<ov::IRemoteContext>> remote_contexts = {gpu_ctx, npu_ctx};

    auto buffer = std::make_shared<ov::npuw::SharedContextBuffer>(byte_size, std::move(remote_contexts), alignment);
    auto descriptor = buffer->get_descriptor();

    ASSERT_NE(descriptor, nullptr);
    auto shared_descriptor = std::dynamic_pointer_cast<ov::SharedContextBufferDescriptor>(descriptor);
    ASSERT_NE(shared_descriptor, nullptr);

    EXPECT_GT(shared_descriptor->get_id(), 0u);
    EXPECT_EQ(shared_descriptor->get_offset(), 0u);

    auto source_buffer = shared_descriptor->get_source_buffer();
    ASSERT_NE(source_buffer, nullptr);
    EXPECT_EQ(source_buffer->get_ptr(), buffer->get_ptr());
    EXPECT_EQ(source_buffer->size(), buffer->size());

    const auto& remote_context_map = shared_descriptor->get_remote_contexts();
    ASSERT_EQ(remote_context_map.size(), 2u);

    const auto gpu_it = remote_context_map.find(gpu_ctx);
    const auto npu_it = remote_context_map.find(npu_ctx);
    ASSERT_NE(gpu_it, remote_context_map.end());
    ASSERT_NE(npu_it, remote_context_map.end());
    ASSERT_NE(gpu_it->second, nullptr);
    ASSERT_NE(npu_it->second, nullptr);
    EXPECT_EQ(gpu_it->second->get_device_name().rfind("GPU", 0), 0u);
    EXPECT_EQ(npu_it->second->get_device_name().rfind("NPU", 0), 0u);
}

}  // namespace


