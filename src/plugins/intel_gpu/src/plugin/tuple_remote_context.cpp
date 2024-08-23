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

TupleRemoteContextImpl::TupleRemoteContextImpl(std::map<std::string, RemoteContextImpl::Ptr> contexts) {
    m_contexts = contexts;
}

const ov::AnyMap& TupleRemoteContextImpl::get_property() const {
    return m_contexts.begin()->second->get_property();
}

std::shared_ptr<TupleRemoteContextImpl> TupleRemoteContextImpl::get_this_shared_ptr() {
    return std::static_pointer_cast<TupleRemoteContextImpl>(shared_from_this());
}

ov::SoPtr<ov::ITensor> TupleRemoteContextImpl::create_host_tensor(const ov::element::Type type, const ov::Shape& shape) {
    OPENVINO_THROW_NOT_IMPLEMENTED("Not Implemented");
}

ov::SoPtr<ov::IRemoteTensor> TupleRemoteContextImpl::create_tensor(const ov::element::Type& type, const ov::Shape& shape, const ov::AnyMap& params) {
    std::vector<ov::SoPtr<ov::IRemoteTensor>> tensors;
    for (auto& item : m_contexts) {
        // std::cout << item.first << std::endl;
        auto a = item.second->create_tensor(type, shape, params);
        tensors.emplace_back(a);
    }
    // std::cout << "tupe tensors size: " << tensors.size() << std::endl;
    return std::make_shared<ov::intel_gpu::TupleRemoteTensorImpl>(get_this_shared_ptr(), tensors);
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
}  // namespace intel_gpu
}  // namespace ov
