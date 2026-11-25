// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/remote_context.hpp"

#include <memory>

#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/usm_host_tensor.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov::intel_gpu {

RemoteContextImpl::RemoteContextImpl(const std::string& device_name, std::vector<cldnn::device::ptr> devices, bool initialize_ctx)
    : m_device_name(device_name) {
    OPENVINO_ASSERT(devices.size() == 1, "[GPU] Currently context can be created for single device only");
    m_device = devices.front();
    init_default_context_type();

    if (initialize_ctx) {
        initialize();
    }
}

cldnn::engine& RemoteContextImpl::get_engine() {
    OPENVINO_ASSERT(m_is_initialized, "[GPU] Can't obtain engine from uninitialized RemoteContext. Please initialize the context first");
    return *m_engine;
}

const cldnn::engine& RemoteContextImpl::get_engine() const {
    OPENVINO_ASSERT(m_is_initialized, "[GPU] Can't obtain engine from uninitialized RemoteContext. Please initialize the context first");
    return *m_engine;
}

const ov::AnyMap& RemoteContextImpl::get_property() const {
    OPENVINO_ASSERT(m_is_initialized, "[GPU] get_property() called on uninitialized context. Please initialize the context before use");
    return properties;
}

std::shared_ptr<RemoteContextImpl> RemoteContextImpl::get_this_shared_ptr() {
    return std::static_pointer_cast<RemoteContextImpl>(shared_from_this());
}

ov::SoPtr<ov::ITensor> RemoteContextImpl::create_host_tensor(const ov::element::Type type, const ov::Shape& shape) {
    OPENVINO_ASSERT(m_is_initialized, "[GPU] create_host_tensor() called on uninitialized context. Please initialize the context before use");
    if (type != ov::element::string && m_engine->use_unified_shared_memory()) {
        return { std::make_shared<USMHostTensor>(get_this_shared_ptr(), type, shape), nullptr };
    }
    return {ov::make_tensor(type, shape), nullptr};
}

ov::SoPtr<ov::IRemoteTensor> RemoteContextImpl::create_tensor(const ov::element::Type& type, const ov::Shape& shape, const ov::AnyMap& params) {
    OPENVINO_ASSERT(m_is_initialized, "[GPU] create_tensor() called on uninitialized context. Please initialize the context before use");

    if (params.empty()) {
        // user wants plugin to allocate tensor by itself and return handle
        return { create_buffer(type, shape), nullptr };
    }
    return create_tensor_from_params(type, shape, params);
}

const std::string& RemoteContextImpl::get_device_name() const {
    return m_device_name;
}

cldnn::memory::ptr RemoteContextImpl::try_get_cached_memory(size_t hash) {
    std::lock_guard<std::mutex> lock(m_cache_mutex);
    if (!m_memory_cache.has(hash)) {
        return nullptr;
    }

    auto cached_memory = m_memory_cache.get(hash);
    if (auto memory = cached_memory.lock()) {
        return memory;
    }

    // The last tensor which wrapped that memory is gone, so the entry is erased instead of being kept as the most recently used one
    m_memory_cache.erase(hash);
    return nullptr;
}

void RemoteContextImpl::add_to_cache(size_t hash, cldnn::memory::ptr memory) {
    std::lock_guard<std::mutex> lock(m_cache_mutex);
    m_memory_cache.add(hash, memory);
}

std::shared_ptr<ov::IRemoteTensor> RemoteContextImpl::create_buffer(const ov::element::Type type, const ov::Shape& shape) {
    return std::make_shared<RemoteTensorImpl>(get_this_shared_ptr(), shape, type, TensorType::BT_BUF_INTERNAL);
}

void RemoteContextImpl::initialize() {
    std::call_once(m_initialize_flag, [this]() {
        GPU_DEBUG_INFO << "Initialize RemoteContext for " << m_device_name << " (" << m_device->get_info().dev_name << ")" << std::endl;

        m_device->initialize();  // Initialize associated device before use
        m_engine = cldnn::engine::create(
            cldnn::get_default_engine_type(), cldnn::get_default_runtime_type(), m_device);

        init_properties();

        m_is_initialized = true;
});
}

}  // namespace ov::intel_gpu
