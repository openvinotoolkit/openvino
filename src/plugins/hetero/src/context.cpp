// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/make_tensor.hpp"
#include "context.hpp"
#include <memory>
#include "remote_tensor.hpp"

namespace ov {
namespace hetero {
// namespace {

// template <typename Type>
// Type extract_object(const ov::AnyMap& params, const ov::Property<Type>& p) {
//     auto itrHandle = params.find(p.name());
//     OPENVINO_ASSERT(itrHandle != params.end(), "[GPU] No parameter ", p.name(), " found in parameters map");
//     ov::Any res = itrHandle->second;
//     return res.as<Type>();
// }

// }  // namespace

HeteroContext::HeteroContext(std::map<std::string, ov::SoPtr<ov::IRemoteContext>> contexts) {
    m_contexts = contexts;
}

const ov::AnyMap& HeteroContext::get_property() const {
    return m_contexts.begin()->second->get_property();
}

std::shared_ptr<HeteroContext> HeteroContext::get_this_shared_ptr() {
    return std::static_pointer_cast<HeteroContext>(shared_from_this());
}

// ov::SoPtr<ov::ITensor> HeteroContext::create_host_tensor(const ov::element::Type type, const ov::Shape& shape) {
//     OPENVINO_THROW_NOT_IMPLEMENTED("Not Implemented");
// }

ov::SoPtr<ov::IRemoteTensor> HeteroContext::create_tensor(const ov::element::Type& type, const ov::Shape& shape, const ov::AnyMap& params) {
    ov::Shape sub_shape;
    for (auto item : shape) {
        sub_shape.emplace_back(item);
    }
    // Only for vllm now
    int head_num = shape[1];
    sub_shape[1] = head_num / m_contexts.size();
    std::vector<ov::SoPtr<ov::IRemoteTensor>> tensors;
    for (auto& item : m_contexts) {
        auto a = item.second->create_tensor(type, sub_shape, params);
        tensors.emplace_back(a);
    }
    return std::make_shared<ov::hetero::HeteroRemoteTensor>(get_this_shared_ptr(), tensors);
}

// const std::string& HeteroContext::get_device_name() const {
//     return m_device_name;
// }

// cldnn::memory::ptr HeteroContext::try_get_cached_memory(size_t hash) {
//     std::lock_guard<std::mutex> lock(m_cache_mutex);
//     if (m_memory_cache.has(hash))
//         return m_memory_cache.get(hash);

//     return nullptr;
// }

// void HeteroContext::add_to_cache(size_t hash, cldnn::memory::ptr memory) {
//     std::lock_guard<std::mutex> lock(m_cache_mutex);
//     m_memory_cache.add(hash, memory);
// }
}  // namespace intel_gpu
}  // namespace ov
