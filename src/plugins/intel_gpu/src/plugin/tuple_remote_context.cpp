// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/tuple_remote_context.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/tuple_remote_tensor.hpp"
#include "intel_gpu/plugin/usm_host_tensor.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/device_query.hpp"
#include <memory>

namespace ov {
namespace intel_gpu {

namespace {

template <typename Type>
Type extract_object(const ov::AnyMap& params, const ov::Property<Type>& p) {
    auto itrHandle = params.find(p.name());
    OPENVINO_ASSERT(itrHandle != params.end(), "[GPU] No parameter ", p.name(), " found in parameters map");
    ov::Any res = itrHandle->second;
    return res.as<Type>();
}

}  // namespace

TupleRemoteContextImpl::TupleRemoteContextImpl(const std::string& device_name, std::vector<cldnn::device::ptr> devices) : m_device_name(device_name) {
    OPENVINO_ASSERT(devices.size() == 1, "[GPU] Currently context can be created for single device only");
    const auto engine_type = cldnn::engine_types::ocl;
    const auto runtime_type = cldnn::runtime_types::ocl;

    m_engine = cldnn::engine::create(engine_type, runtime_type, devices.front());

    GPU_DEBUG_LOG << "Initialize RemoteContext for " << m_device_name << " (" << m_engine->get_device_info().dev_name << ")" << std::endl;
    init_properties();
}

TupleRemoteContextImpl::TupleRemoteContextImpl(std::map<std::string, RemoteContextImpl::Ptr> contexts) {
    std::cout << "create tuple context\n";
    m_contexts = contexts;
}

void TupleRemoteContextImpl::init_properties() {
    properties = { ov::intel_gpu::ocl_context(m_engine->get_user_context()) };

    switch (m_type) {
    case ContextType::OCL:
        properties.insert(ov::intel_gpu::context_type(ov::intel_gpu::ContextType::OCL));
        properties.insert(ov::intel_gpu::ocl_queue(m_external_queue));
        break;
    case ContextType::VA_SHARED:
        properties.insert(ov::intel_gpu::context_type(ov::intel_gpu::ContextType::VA_SHARED));
        properties.insert(ov::intel_gpu::va_device(m_va_display));
        break;
    default:
        OPENVINO_THROW("[GPU] Unsupported shared context type ", m_type);
    }
}

const ov::AnyMap& TupleRemoteContextImpl::get_property() const {
    return properties;
}

std::shared_ptr<TupleRemoteContextImpl> TupleRemoteContextImpl::get_this_shared_ptr() {
    return std::static_pointer_cast<TupleRemoteContextImpl>(shared_from_this());
}

ov::SoPtr<ov::ITensor> TupleRemoteContextImpl::create_host_tensor(const ov::element::Type type, const ov::Shape& shape) {
    // if (m_engine->use_unified_shared_memory()) {
    //     return { std::make_shared<USMHostTensor>(get_this_shared_ptr(), type, shape), nullptr };
    // } else {
        return { ov::make_tensor(type, shape), nullptr };
    // }
}

ov::SoPtr<ov::IRemoteTensor> TupleRemoteContextImpl::create_tensor(const ov::element::Type& type, const ov::Shape& shape, const ov::AnyMap& params) {
//     std::cout << "********* create remote tensor ******* " << this->get_device_name() << std::endl;
//     if (params.empty()) {
//         std::cout << "params.empty()\n";
//         // user wants plugin to allocate tensor by itself and return handle
//         if (this->get_device_name() == "GPU.-1") {
//             std::cout << "virtual device create tensor\n";
//             return { create_buffer(type, shape, true), nullptr };
//         }
    std::vector<ov::SoPtr<ov::IRemoteTensor>> tensors;
    for (auto& item : m_contexts) {
        if (item.first == "0") {
            continue;
        }
        std::cout << item.first << std::endl;
        auto a = item.second->create_tensor(type, shape, params);
        tensors.emplace_back(a);
    }
    std::cout << "tupe tensors size: " << tensors.size() << std::endl;
    return std::make_shared<ov::intel_gpu::TupleRemoteTensorImpl>(get_this_shared_ptr(), tensors);
        // return { create_buffer(type, shape), nullptr };
//     } else {
//         std::cout << "params not empty()\n";
//         // user will supply shared object handle
//         auto mem_type = extract_object(params, ov::intel_gpu::shared_mem_type);

//         bool is_usm = mem_type == ov::intel_gpu::SharedMemType::USM_HOST_BUFFER ||
//                       mem_type == ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER ||
//                       mem_type == ov::intel_gpu::SharedMemType::USM_USER_BUFFER;

//         OPENVINO_ASSERT(!is_usm || m_engine->use_unified_shared_memory(),
//                         "[GPU] Can't create USM tensor as USM is not supported (or manually disabled) on current device");

//         if (ov::intel_gpu::SharedMemType::VA_SURFACE == mem_type) {
//             check_if_shared();
//             return { reuse_surface(type, shape, params), nullptr };
//         } else if (ov::intel_gpu::SharedMemType::USM_HOST_BUFFER == mem_type) {
//             return { create_usm(type, shape, TensorType::BT_USM_HOST_INTERNAL), nullptr };
//         } else if (ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER == mem_type) {
//             return { create_usm(type, shape, TensorType::BT_USM_DEVICE_INTERNAL), nullptr };
//         } else {
//             TensorType tensor_type;
//             cldnn::shared_handle mem = nullptr;

//             if (ov::intel_gpu::SharedMemType::OCL_BUFFER == mem_type) {
//                 tensor_type = TensorType::BT_BUF_SHARED;
//                 mem = extract_object(params, ov::intel_gpu::mem_handle);
//             } else if (ov::intel_gpu::SharedMemType::USM_USER_BUFFER == mem_type) {
//                 tensor_type = TensorType::BT_USM_SHARED;
//                 mem = extract_object(params, ov::intel_gpu::mem_handle);
//             } else if (ov::intel_gpu::SharedMemType::OCL_IMAGE2D == mem_type) {
//                 tensor_type = TensorType::BT_IMG_SHARED;
//                 mem = extract_object(params, ov::intel_gpu::mem_handle);
// #ifdef _WIN32
//             } else if (ov::intel_gpu::SharedMemType::DX_BUFFER == mem_type) {
//                 tensor_type = TensorType::BT_DX_BUF_SHARED;
//                 mem = extract_object(params, ov::intel_gpu::dev_object_handle);
//                 check_if_shared();
// #endif
//             } else {
//                 OPENVINO_THROW("[GPU] Unsupported shared object type ", mem_type);
//             }

//             return { reuse_memory(type, shape, mem, tensor_type), nullptr };
//         }
//     }
}

// For external contexts we try to match underlying handles with default contexts created by plugin to find device name
std::string TupleRemoteContextImpl::get_device_name(const std::map<std::string, RemoteContextImpl::Ptr>& known_contexts,
                                               const cldnn::device::ptr current_device) const {
    std::string device_name = "GPU";
    for (auto& c : known_contexts) {
        if (c.second->get_engine().get_device()->is_same(current_device)) {
            device_name = c.second->get_device_name();
            break;
        }
    }
    return device_name;
}

const std::string& TupleRemoteContextImpl::get_device_name() const {
    return m_device_name;
}

cldnn::memory::ptr TupleRemoteContextImpl::try_get_cached_memory(size_t hash) {
    std::lock_guard<std::mutex> lock(m_cache_mutex);
    if (m_memory_cache.has(hash))
        return m_memory_cache.get(hash);

    return nullptr;
}

void TupleRemoteContextImpl::add_to_cache(size_t hash, cldnn::memory::ptr memory) {
    std::lock_guard<std::mutex> lock(m_cache_mutex);
    m_memory_cache.add(hash, memory);
}

// std::shared_ptr<ov::IRemoteTensor> TupleRemoteContextImpl::reuse_surface(const ov::element::Type type, const ov::Shape& shape, const ov::AnyMap& params) {
//     uint32_t plane = extract_object(params, ov::intel_gpu::va_plane);

// #ifdef _WIN32
//     cldnn::shared_handle surf = extract_object(params, ov::intel_gpu::dev_object_handle);
//     return std::make_shared<RemoteTensorImpl>(get_this_shared_ptr(), shape, type, TensorType::BT_SURF_SHARED, surf, 0, plane);
// #else
//     cldnn::shared_surface surf = extract_object(params, ov::intel_gpu::dev_object_handle);
//     return std::make_shared<RemoteTensorImpl>(get_this_shared_ptr(), shape, type, TensorType::BT_SURF_SHARED, nullptr, surf, plane);
// #endif
// }

std::shared_ptr<ov::IRemoteTensor> TupleRemoteContextImpl::reuse_memory(const ov::element::Type type,
                                                                   const ov::Shape& shape,
                                                                   cldnn::shared_handle mem,
                                                                   TensorType tensor_type) {
    // return std::make_shared<RemoteTensorImpl>(get_this_shared_ptr(), shape, type, tensor_type, mem);
    return nullptr;
}

std::shared_ptr<ov::IRemoteTensor> TupleRemoteContextImpl::create_buffer(const ov::element::Type type, const ov::Shape& shape, bool is_virtual) {
    // return std::make_shared<RemoteTensorImpl>(get_this_shared_ptr(), shape, type, TensorType::BT_BUF_INTERNAL, nullptr, 0, 0, is_virtual);
    return nullptr;
}

std::shared_ptr<ov::IRemoteTensor> TupleRemoteContextImpl::create_usm(const ov::element::Type type, const ov::Shape& shape, TensorType alloc_type) {
    // return std::make_shared<RemoteTensorImpl>(get_this_shared_ptr(), shape, type, alloc_type);
    return nullptr;
}

void TupleRemoteContextImpl::check_if_shared() const {
    OPENVINO_ASSERT(m_type == ContextType::VA_SHARED, "[GPU] Shared context is required to to share this type of memory");
}

}  // namespace intel_gpu
}  // namespace ov
