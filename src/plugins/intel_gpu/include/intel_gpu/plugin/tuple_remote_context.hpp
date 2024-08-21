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
#include "intel_gpu/plugin/remote_context.hpp"

#include <string>
#include <map>
#include <memory>
#include <atomic>

namespace ov {
namespace intel_gpu {
class RemoteContextImpl;
class TupleRemoteContextImpl : public ov::IRemoteContext {
public:
    using Ptr = std::shared_ptr<TupleRemoteContextImpl>;

    // TupleRemoteContextImpl(std::map<std::string, RemoteContextImpl::Ptr> contexts);
    TupleRemoteContextImpl(std::map<std::string, RemoteContextImpl::Ptr> contexts);

    const std::string& get_device_name() const override;

    const ov::AnyMap& get_property() const override;
    ov::SoPtr<ov::ITensor> create_host_tensor(const ov::element::Type type, const ov::Shape& shape) override;
    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type, const ov::Shape& shape, const ov::AnyMap& params) override;

    cldnn::memory::ptr try_get_cached_memory(size_t hash);
    void add_to_cache(size_t hash, cldnn::memory::ptr memory);

private:
    std::shared_ptr<TupleRemoteContextImpl> get_this_shared_ptr();
    std::string m_device_name = "VIRTUAL_DEVICE";
    static const size_t cache_capacity = 100;
    cldnn::LruCache<size_t, cldnn::memory::ptr> m_memory_cache = cldnn::LruCache<size_t, cldnn::memory::ptr>(cache_capacity);
    std::mutex m_cache_mutex;

    std::map<std::string, RemoteContextImpl::Ptr> m_contexts;
};

}  // namespace intel_gpu
}  // namespace ov
