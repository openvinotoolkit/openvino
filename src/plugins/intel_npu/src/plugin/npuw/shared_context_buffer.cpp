// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>

#include "shared_context_buffer.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"

namespace {
std::atomic_size_t g_shared_context_buffer_id{1};
}

namespace {

ov::SoPtr<ov::IRemoteTensor> bind_gpu_remote_ctx(ov::AlignedBuffer& buffer, size_t aligned_size,
                                                 const ov::SoPtr<ov::IRemoteContext>& remote_ctx) {
    OPENVINO_ASSERT(buffer.get_ptr(), "Cannot bind_gpu_remote_ctx: buffer is null");
    OPENVINO_ASSERT(remote_ctx, "Cannot bind_gpu_remote_ctx: remote context is null");

    ov::SoPtr<ov::IRemoteTensor> gpu_remote_tensor;
    try {
        ov::AnyMap params = {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::CPU_VA},
                             {ov::intel_gpu::cpu_va.name(), buffer.get_ptr()},
                             {ov::intel_gpu::cpu_va_size.name(), aligned_size}}; 
    
        gpu_remote_tensor = remote_ctx->create_tensor(ov::element::u8, ov::Shape{buffer.size()}, params);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Cannot bind_gpu_remote_ctx: failed to create remote tensor: ") + e.what());
    }
    return gpu_remote_tensor;
}

ov::SoPtr<ov::IRemoteTensor> bind_npu_remote_ctx(ov::AlignedBuffer& buffer,
                                                size_t aligned_size,
                                                 const ov::SoPtr<ov::IRemoteContext>& remote_ctx) {
    OPENVINO_ASSERT(buffer.get_ptr(), "Cannot bind_npu_remote_ctx: buffer is null");
    OPENVINO_ASSERT(remote_ctx, "Cannot bind_npu_remote_ctx: remote context is null");
    
    // TODO user real shape and element type for the remote tensor, for now we use u8 and shape of {buffer.size()}
    try {
        ov::AnyMap params = {
            {ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::CPU_VA},
            {ov::intel_npu::mem_handle.name(), buffer.get_ptr()},
            {ov::intel_npu::tensor_type.name(), ov::intel_npu::TensorType::INPUT},
        };
        return remote_ctx->create_tensor(ov::element::u8, ov::Shape{aligned_size}, params);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Cannot bind_npu_remote_ctx: failed to create remote tensor: ") + e.what());
    }
}
}

ov::npuw::SharedContextBuffer::SharedContextBuffer(size_t byte_size,
                                                   std::vector<ov::SoPtr<ov::IRemoteContext>> remote_contexts,
                                                   size_t alignment)
    : ov::AlignedBuffer(((byte_size + alignment - 1) / alignment) * alignment, alignment),
            m_id(g_shared_context_buffer_id.fetch_add(1, std::memory_order_relaxed)) {
    OPENVINO_ASSERT(remote_contexts.size() > 0, "SharedContextBuffer: remote_contexts must not be empty");
    m_real_buffer_size = this->size();    // Set the actual byte size, as AlignedBuffer constructor may have rounded it up to alignment
    memset(m_aligned_buffer, 0, m_real_buffer_size); // Fill by zeros

    m_byte_size = byte_size;     // Set the actual constant size to AlignedBuffer, not aligned
    for (const auto& remote_ctx : remote_contexts) {
        auto remote_tensor = bind_remote_context(remote_ctx);
        if (remote_tensor) {
            m_remote_contexts.emplace(remote_ctx, remote_tensor);
        }
    }
}

ov::npuw::SharedContextBuffer::~SharedContextBuffer() = default;

size_t ov::npuw::SharedContextBuffer::get_real_buffer_size() const {
    return m_real_buffer_size;
}

std::shared_ptr<ov::IBufferDescriptor> ov::npuw::SharedContextBuffer::get_descriptor() const {
    auto self = std::const_pointer_cast<SharedContextBuffer>(shared_from_this());
    return std::make_shared<SharedContextBufferDescriptor>(
        m_id,
        0,
        m_real_buffer_size,
        std::static_pointer_cast<ov::AlignedBuffer>(std::move(self)),
        m_remote_contexts);
}

ov::SoPtr<ov::IRemoteTensor> ov::npuw::SharedContextBuffer::get_remote_tensors_if_exist(ov::SoPtr<ov::IRemoteContext> remote_ctx) const {
    // TODO maybe use device name instead of remote_ctx pointer for lookup? since remote_ctx pointer may be different for the same device (or not?)
    auto it = m_remote_contexts.find(remote_ctx);
    if (it != m_remote_contexts.end()) {
        return it->second;
    }
    return nullptr;
}

ov::SoPtr<ov::IRemoteTensor> ov::npuw::SharedContextBuffer::bind_remote_context(const ov::SoPtr<ov::IRemoteContext>& remote_ctx) {
    if (!remote_ctx) {
        return nullptr;
    }

    const auto& device_name = remote_ctx->get_device_name();
    if (device_name.find("GPU") != std::string::npos) {
        return bind_gpu_remote_ctx(*this, get_real_buffer_size(), remote_ctx);
    } else if (device_name.find("NPU") != std::string::npos) {
        return bind_npu_remote_ctx(*this, get_real_buffer_size(), remote_ctx);
    } else {
        OPENVINO_THROW("Unsupported remote context device: " + device_name);
    }
}
