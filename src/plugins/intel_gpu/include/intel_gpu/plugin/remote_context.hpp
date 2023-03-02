// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/lru_cache.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include <ie_parameter.hpp>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <blob_factory.hpp>
#include <ie_remote_context.hpp>

#ifndef NOMINMAX
# define NOMINMAX
#endif

#ifdef _WIN32
# include <gpu/gpu_context_api_dx.hpp>
#else
# include <gpu/gpu_context_api_va.hpp>
#endif

#include <string>
#include <map>
#include <memory>
#include <atomic>

namespace ov {
namespace intel_gpu {

enum class BlobType {
    BT_EMPTY,
    BT_BUF_INTERNAL,
    BT_BUF_SHARED,
    BT_USM_SHARED,
    BT_USM_HOST_INTERNAL,
    BT_USM_DEVICE_INTERNAL,
    BT_IMG_SHARED,
    BT_SURF_SHARED,
    BT_DX_BUF_SHARED,
};

template <typename Result>
Result extract_object(const InferenceEngine::ParamMap& params, const std::string& key) {
    auto itrHandle = params.find(key);
    OPENVINO_ASSERT(itrHandle != params.end(), "[GPU] No parameter ", key, " found in ParamsMap");
    return itrHandle->second.as<Result>();
}

class RemoteContextImpl {
public:
    enum ContextType {
        OCL,
        DEV_SHARED
    };

    using Ptr = std::shared_ptr<RemoteContextImpl>;
    using CPtr = std::shared_ptr<const RemoteContextImpl>;

    RemoteContextImpl(std::string device_name, std::vector<cldnn::device::ptr> devices);
    RemoteContextImpl(const std::vector<RemoteContextImpl::Ptr>& known_contexts, const InferenceEngine::ParamMap& params);

    InferenceEngine::ParamMap get_params() const;
    std::string get_device_name() const noexcept;
    InferenceEngine::MemoryBlob::Ptr create_host_blob(InferenceEngine::gpu::ClContext::Ptr public_context, const InferenceEngine::TensorDesc& desc);
    InferenceEngine::RemoteBlob::Ptr create_blob(InferenceEngine::gpu::ClContext::Ptr public_context,
                                                 const InferenceEngine::TensorDesc& desc,
                                                 const InferenceEngine::ParamMap& params = {});

    cldnn::engine& get_engine() { return *m_engine; }
    InferenceEngine::gpu_handle_param get_external_queue() const { return m_external_queue; }

    cldnn::memory::ptr try_get_cached_memory(size_t hash);
    void add_to_cache(size_t hash, cldnn::memory::ptr memory);

private:
    std::string get_device_name(const std::vector<RemoteContextImpl::Ptr>& known_contexts,
                                const cldnn::device::ptr current_device);
    InferenceEngine::RemoteBlob::Ptr reuse_surface(InferenceEngine::gpu::ClContext::Ptr public_context,
                                                   const InferenceEngine::TensorDesc& desc,
                                                   const InferenceEngine::ParamMap& params);
    InferenceEngine::RemoteBlob::Ptr reuse_memory(InferenceEngine::gpu::ClContext::Ptr public_context,
                                                  const InferenceEngine::TensorDesc& desc,
                                                  cldnn::shared_handle mem,
                                                  BlobType blob_type);
    InferenceEngine::RemoteBlob::Ptr create_buffer(InferenceEngine::gpu::ClContext::Ptr public_context, const InferenceEngine::TensorDesc& desc);
    InferenceEngine::RemoteBlob::Ptr create_usm(InferenceEngine::gpu::ClContext::Ptr public_context,
                                                const InferenceEngine::TensorDesc& desc,
                                                BlobType alloc_type);
    void check_if_shared();

    std::shared_ptr<cldnn::engine> m_engine;
    InferenceEngine::gpu_handle_param m_va_display;
    InferenceEngine::gpu_handle_param m_external_queue;
    static const size_t cache_capacity = 100;

    ContextType m_type;
    std::string m_device_name = "";
    const std::string m_plugin_name;
    cldnn::LruCache<size_t, cldnn::memory::ptr> m_memory_cache;
    std::mutex m_cache_mutex;
};

// Template class below is needed to allow proper cast of user contexts
// We have the following public classes hierarchy:
//        RemoteContext
//              |
//          ClContext
//        |          |
//   VAContext      D3DContext
// So our implementation must allow casting of context object to proper type user type (ClContext, VAContext or D3DContext)
// Thus we introduce this template which have 3 instances with different base classes:
//                RemoteContext
//                      |
//        ---------- ClContext -----------
//        |             |                |
//   VAContext          |            D3DContext
//        |             |                |
// RemoteVAContext  RemoteCLContext  RemoteD3DContext
//
// All these context types are just thin wrappers that calls common context internal impl (RemoteContextImpl)
template<typename PublicContextType>
class TypedRemoteContext : public PublicContextType {
public:
    using Ptr = std::shared_ptr<TypedRemoteContext>;

    TypedRemoteContext(std::string device_name, std::vector<cldnn::device::ptr> devices)
        : m_impl(std::make_shared<RemoteContextImpl>(device_name, devices)) {}
    TypedRemoteContext(const std::vector<RemoteContextImpl::Ptr>& known_contexts, const InferenceEngine::ParamMap& params)
        : m_impl(std::make_shared<RemoteContextImpl>(known_contexts, params)) {}

    InferenceEngine::ParamMap getParams() const override { return m_impl->get_params(); }
    std::string getDeviceName() const noexcept override { return m_impl->get_device_name(); }
    InferenceEngine::MemoryBlob::Ptr CreateHostBlob(const InferenceEngine::TensorDesc& desc) override {
        return m_impl->create_host_blob(std::dynamic_pointer_cast<InferenceEngine::gpu::ClContext>(this->shared_from_this()), desc);
    }
    InferenceEngine::RemoteBlob::Ptr CreateBlob(const InferenceEngine::TensorDesc& desc, const InferenceEngine::ParamMap& params = {}) override {
        return m_impl->create_blob(std::dynamic_pointer_cast<InferenceEngine::gpu::ClContext>(this->shared_from_this()), desc, params);
    }

    RemoteContextImpl::Ptr get_impl() { return m_impl; }

private:
    std::shared_ptr<RemoteContextImpl> m_impl;
};

using RemoteCLContext = TypedRemoteContext<InferenceEngine::gpu::ClContext>;
#ifdef _WIN32
using RemoteD3DContext = TypedRemoteContext<InferenceEngine::gpu::D3DContext>;
#else
using RemoteVAContext = TypedRemoteContext<InferenceEngine::gpu::VAContext>;
#endif

inline std::shared_ptr<RemoteContextImpl> get_context_impl(InferenceEngine::gpu::ClContext::Ptr context) {
    OPENVINO_ASSERT(context != nullptr, "[GPU] Couldn't get impl from invalid context object");
#ifdef _WIN32
    if (auto ptr = context->as<RemoteD3DContext>())
        return ptr->get_impl();
#else
    if (auto ptr = context->as<RemoteVAContext>())
        return ptr->get_impl();
#endif
    if (auto ptr = context->as<RemoteCLContext>())
        return ptr->get_impl();

    OPENVINO_ASSERT(false, "[GPU] Couldn't get context impl from public context object.");
}

inline std::shared_ptr<RemoteContextImpl> get_context_impl(InferenceEngine::RemoteContext::Ptr context) {
    OPENVINO_ASSERT(context != nullptr, "[GPU] Couldn't get impl from invalid context object");
    auto casted = std::dynamic_pointer_cast<InferenceEngine::gpu::ClContext>(context);
    OPENVINO_ASSERT(casted != nullptr, "[GPU] Couldn't get context impl: Context type is not ClContext or it's derivatives");
    return get_context_impl(casted);
}

}  // namespace intel_gpu
}  // namespace ov
