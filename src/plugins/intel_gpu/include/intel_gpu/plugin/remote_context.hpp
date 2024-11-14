// Copyright (C) 2018-2024 Intel Corporation
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
#include <atomic>

namespace ov {
namespace intel_gpu {

class RemoteContextImpl : public ov::IRemoteContext {
public:
    using Ptr = std::shared_ptr<RemoteContextImpl>;

    RemoteContextImpl(const std::string& device_name, std::vector<cldnn::device::ptr> devices);
    RemoteContextImpl(const std::map<std::string, RemoteContextImpl::Ptr>& known_contexts, const ov::AnyMap& params);

    const std::string& get_device_name() const override;

    const ov::AnyMap& get_property() const override;
    ov::SoPtr<ov::ITensor> create_host_tensor(const ov::element::Type type, const ov::Shape& shape) override;
    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type, const ov::Shape& shape, const ov::AnyMap& params) override;

    cldnn::engine& get_engine() { return *m_engine; }
    ov::intel_gpu::gpu_handle_param get_external_queue() const { return m_external_queue; }

    cldnn::memory::ptr try_get_cached_memory(size_t hash);
    void add_to_cache(size_t hash, cldnn::memory::ptr memory);

private:
    std::shared_ptr<RemoteContextImpl> get_this_shared_ptr();

    std::string get_device_name(const std::map<std::string, RemoteContextImpl::Ptr>& known_contexts, const cldnn::device::ptr current_device) const;
    std::shared_ptr<ov::IRemoteTensor> reuse_surface(const ov::element::Type type, const ov::Shape& shape, const ov::AnyMap& params);
    std::shared_ptr<ov::IRemoteTensor> reuse_memory(const ov::element::Type type, const ov::Shape& shape, cldnn::shared_handle mem, TensorType tensor_type);
    std::shared_ptr<ov::IRemoteTensor> create_buffer(const ov::element::Type type, const ov::Shape& shape);
    std::shared_ptr<ov::IRemoteTensor> create_usm(const ov::element::Type type, const ov::Shape& shape, TensorType alloc_type);
    void check_if_shared() const;

    void init_properties();

    std::shared_ptr<cldnn::engine> m_engine;
    ov::intel_gpu::gpu_handle_param m_va_display = nullptr;
    ov::intel_gpu::gpu_handle_param m_external_queue = nullptr;

    ContextType m_type = ContextType::OCL;
    std::string m_device_name = "";
    static const size_t cache_capacity = 100;
    cldnn::LruCache<size_t, cldnn::memory::ptr> m_memory_cache = cldnn::LruCache<size_t, cldnn::memory::ptr>(cache_capacity);
    std::mutex m_cache_mutex;

    ov::AnyMap properties;
};

inline RemoteContextImpl::Ptr get_context_impl(ov::SoPtr<ov::IRemoteContext> ptr) {
    auto casted = std::dynamic_pointer_cast<RemoteContextImpl>(ptr._ptr);
    OPENVINO_ASSERT(casted, "[GPU] Invalid remote context type. Can't cast to ov::intel_gpu::RemoteContext type");
    return casted;
}

}  // namespace intel_gpu
}  // namespace ov
