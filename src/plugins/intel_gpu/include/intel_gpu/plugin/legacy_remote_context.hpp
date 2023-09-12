// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/lru_cache.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/legacy_remote_blob.hpp"

#include <ie_parameter.hpp>
#include <blob_factory.hpp>
#include <ie_remote_context.hpp>
#include <remote_utils.hpp>

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

    explicit TypedRemoteContext(std::shared_ptr<RemoteContextImpl> impl) : m_impl(impl) {}
    TypedRemoteContext(std::string device_name, std::vector<cldnn::device::ptr> devices)
        : m_impl(std::make_shared<RemoteContextImpl>(device_name, devices)) {}
    TypedRemoteContext(const std::map<std::string, RemoteContextImpl::Ptr>& known_contexts, const InferenceEngine::ParamMap& params)
        : m_impl(std::make_shared<RemoteContextImpl>(known_contexts, params)) {}

    InferenceEngine::ParamMap getParams() const override { return m_impl->get_property(); }
    std::string getDeviceName() const noexcept override { return m_impl->get_device_name(); }
    InferenceEngine::MemoryBlob::Ptr CreateHostBlob(const InferenceEngine::TensorDesc& desc) override {
        auto new_tensor = m_impl->create_host_tensor(InferenceEngine::details::convertPrecision(desc.getPrecision()), ov::Shape(desc.getDims()));
        return std::dynamic_pointer_cast<InferenceEngine::MemoryBlob>(make_blob_with_precision(desc, new_tensor->data()));
    }
    InferenceEngine::RemoteBlob::Ptr CreateBlob(const InferenceEngine::TensorDesc& desc, const InferenceEngine::ParamMap& params = {}) override {
        auto new_tensor = m_impl->create_tensor(InferenceEngine::details::convertPrecision(desc.getPrecision()), ov::Shape(desc.getDims()), params);
        auto tensor_impl = std::dynamic_pointer_cast<RemoteTensorImpl>(new_tensor._ptr);
        OPENVINO_ASSERT(tensor_impl, "[GPU] Unexpected tensor impl type");
        auto mem_type = tensor_impl->get_properties().at(ov::intel_gpu::shared_mem_type.name()).as<ov::intel_gpu::SharedMemType>();
        if (mem_type == ov::intel_gpu::SharedMemType::OCL_BUFFER) {
            return std::make_shared<RemoteCLbuffer>(tensor_impl, std::dynamic_pointer_cast<InferenceEngine::gpu::ClContext>(this->shared_from_this()));
        } else if (mem_type == ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER ||
            mem_type == ov::intel_gpu::SharedMemType::USM_HOST_BUFFER ||
            mem_type == ov::intel_gpu::SharedMemType::USM_USER_BUFFER) {
            return std::make_shared<RemoteUSMbuffer>(tensor_impl, std::dynamic_pointer_cast<InferenceEngine::gpu::ClContext>(this->shared_from_this()));
        } else if (mem_type == ov::intel_gpu::SharedMemType::OCL_IMAGE2D) {
            return std::make_shared<RemoteCLImage2D>(tensor_impl, std::dynamic_pointer_cast<InferenceEngine::gpu::ClContext>(this->shared_from_this()));
#ifdef _WIN32
        } else if (mem_type == ov::intel_gpu::SharedMemType::DX_BUFFER) {
            return std::make_shared<RemoteD3DBuffer>(tensor_impl, std::dynamic_pointer_cast<InferenceEngine::gpu::ClContext>(this->shared_from_this()));
        } else if (mem_type == ov::intel_gpu::SharedMemType::VA_SURFACE) {
            return std::make_shared<RemoteD3DSurface>(tensor_impl, std::dynamic_pointer_cast<InferenceEngine::gpu::ClContext>(this->shared_from_this()));
#else
        } else if (mem_type == ov::intel_gpu::SharedMemType::VA_SURFACE) {
            return std::make_shared<RemoteVASurface>(tensor_impl, std::dynamic_pointer_cast<InferenceEngine::gpu::ClContext>(this->shared_from_this()));
#endif
        }
        OPENVINO_THROW("[GPU] CreateBlob error: Unsupported memory type: ", mem_type);
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

inline ov::SoPtr<ov::IRemoteContext> wrap_if_old_api(std::shared_ptr<RemoteContextImpl> new_impl, bool is_new_api) {
    if (is_new_api) {
        return new_impl;
    } else {
        auto remote_properties = new_impl->get_property();
        auto context_type = remote_properties.at(ov::intel_gpu::context_type.name()).as<ov::intel_gpu::ContextType>();
        if (context_type == ov::intel_gpu::ContextType::OCL) {
            return ov::legacy_convert::convert_remote_context(std::make_shared<RemoteCLContext>(new_impl));
        } else if (context_type == ov::intel_gpu::ContextType::VA_SHARED) {
    #ifdef _WIN32
            return ov::legacy_convert::convert_remote_context(std::make_shared<RemoteD3DContext>(new_impl));
    #else
            return ov::legacy_convert::convert_remote_context(std::make_shared<RemoteVAContext>(new_impl));
    #endif
        }
    }
    OPENVINO_THROW("[GPU] Unexpected context parameters");
}


inline RemoteContextImpl::Ptr get_context_impl(ov::SoPtr<ov::IRemoteContext> ptr) {
    if (auto wrapper = std::dynamic_pointer_cast<InferenceEngine::IRemoteContextWrapper>(ptr._ptr)) {
        auto legacy_context = wrapper->get_context();
        if (auto legacy_context_impl = std::dynamic_pointer_cast<RemoteCLContext>(legacy_context)) {
            return legacy_context_impl->get_impl();
#ifdef _WIN32
        } else if (auto legacy_context_impl = std::dynamic_pointer_cast<RemoteD3DContext>(legacy_context)) {
            return legacy_context_impl->get_impl();
#else
        } else if (auto legacy_context_impl = std::dynamic_pointer_cast<RemoteVAContext>(legacy_context)) {
            return legacy_context_impl->get_impl();
#endif
        }
    }
    auto casted = std::dynamic_pointer_cast<RemoteContextImpl>(ptr._ptr);
    OPENVINO_ASSERT(casted, "[GPU] Invalid remote context type. Can't cast to ov::intel_gpu::RemoteContext type");
    return casted;
}

}  // namespace intel_gpu
}  // namespace ov
