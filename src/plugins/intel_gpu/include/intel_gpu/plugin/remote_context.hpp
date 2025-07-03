// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef NOMINMAX
# define NOMINMAX
#endif

#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/iremote_context.hpp"

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/lru_cache.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include <string>
#include <map>
#include <memory>

namespace ov::intel_gpu {

class RemoteContextImpl : public ov::IRemoteContext {
public:
    using Ptr = std::shared_ptr<RemoteContextImpl>;

    /**
     * @brief Constructs a RemoteContextImpl for internal plugin use.
     *
     * @param device_name Name of the target device.
     * @param devices List of devices associated with the plugin.
     * @param initialize_ctx If true (default), the context is initialized immediately.
     *                       If false, the context is created in an uninitialized state.
     *
     * Used to serve internal plugin needs, such as creating a context for a specific device.
     * Supports optional delayed initialization.
     */
    RemoteContextImpl(const std::string& device_name, std::vector<cldnn::device::ptr> devices, bool initialize_ctx = true);

    /**
     * @brief Constructs a RemoteContextImpl from a user-provided external context or device.
     *
     * @param known_contexts Map of existing in Plugin contexts.
     * @param params Configuration parameters for the remote context. May include:
     *               context, device handles, target device index, tile index, or an external queue handle.
     *
     * Used to serve external context requests, such as when the user provides an existing context.
     * Always creates an initialized context.
     */
    RemoteContextImpl(const std::map<std::string, RemoteContextImpl::Ptr>& known_contexts, const ov::AnyMap& params);

    const std::string& get_device_name() const override;

    const ov::AnyMap& get_property() const override;
    ov::SoPtr<ov::ITensor> create_host_tensor(const ov::element::Type type, const ov::Shape& shape) override;
    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type, const ov::Shape& shape, const ov::AnyMap& params) override;

    cldnn::engine& get_engine();
    const cldnn::engine& get_engine() const;
    const cldnn::device& get_device() { return *m_device; }
    ov::intel_gpu::gpu_handle_param get_external_queue() const { return m_external_queue; }

    cldnn::memory::ptr try_get_cached_memory(size_t hash);
    void add_to_cache(size_t hash, cldnn::memory::ptr memory);

    /**
     * @brief Initializes the RemoteContext and its associated device.
     *
     * This method performs context initialization if it was deferred during construction
     * (i.e., when the constructor was called with initialize_ctx = false).
     *
     * Has no effect if the context is already initialized.
     */
    void initialize();
    bool is_initialized() const { return m_is_initialized; }

private:
    std::shared_ptr<RemoteContextImpl> get_this_shared_ptr();

    std::string get_device_name(const std::map<std::string, RemoteContextImpl::Ptr>& known_contexts, const cldnn::device::ptr current_device) const;
    std::shared_ptr<ov::IRemoteTensor> reuse_surface(const ov::element::Type type, const ov::Shape& shape, const ov::AnyMap& params);
    std::shared_ptr<ov::IRemoteTensor> reuse_memory(const ov::element::Type type, const ov::Shape& shape, cldnn::shared_handle mem, TensorType tensor_type);
    std::shared_ptr<ov::IRemoteTensor> create_buffer(const ov::element::Type type, const ov::Shape& shape);
    std::shared_ptr<ov::IRemoteTensor> create_usm(const ov::element::Type type, const ov::Shape& shape, TensorType alloc_type);
    void check_if_shared() const;

    void init_properties();

    std::shared_ptr<cldnn::device> m_device;
    std::shared_ptr<cldnn::engine> m_engine;
    ov::intel_gpu::gpu_handle_param m_va_display = nullptr;
    ov::intel_gpu::gpu_handle_param m_external_queue = nullptr;

    ContextType m_type = ContextType::OCL;
    std::string m_device_name = "";
    static const size_t cache_capacity = 100;
    cldnn::LruCache<size_t, cldnn::memory::ptr> m_memory_cache = cldnn::LruCache<size_t, cldnn::memory::ptr>(cache_capacity);
    std::mutex m_cache_mutex;

    bool m_is_initialized = false;
    std::once_flag m_initialize_flag;

    ov::AnyMap properties;
};

inline RemoteContextImpl::Ptr get_context_impl(ov::SoPtr<ov::IRemoteContext> ptr) {
    auto casted = std::dynamic_pointer_cast<RemoteContextImpl>(ptr._ptr);
    OPENVINO_ASSERT(casted, "[GPU] Invalid remote context type. Can't cast to ov::intel_gpu::RemoteContext type");
    return casted;
}

}  // namespace ov::intel_gpu
