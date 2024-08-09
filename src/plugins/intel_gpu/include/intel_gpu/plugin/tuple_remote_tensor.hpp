// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef NOMINMAX
# define NOMINMAX
#endif

#ifdef _WIN32
# include <openvino/runtime/intel_gpu/ocl/dx.hpp>
#else
# include <openvino/runtime/intel_gpu/ocl/va.hpp>
#endif
#include "openvino/runtime/iremote_tensor.hpp"

#include "intel_gpu/runtime/memory_caps.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/tuple_remote_context.hpp"

#include <string>
#include <map>
#include <memory>

namespace ov {
namespace intel_gpu {
class RemoteContextImpl;
class RemoteTensorImpl;

class TupleRemoteTensorImpl : public ov::IRemoteTensor {
    friend class RemoteAllocator;
public:
    TupleRemoteTensorImpl(std::shared_ptr<TupleRemoteContextImpl> context, std::vector<ov::SoPtr<ov::IRemoteTensor>> tensors);

    ~TupleRemoteTensorImpl() override;
    const AnyMap& get_properties() const override;
    const std::string& get_device_name() const override;

    void set_shape(ov::Shape shape) override;
    const ov::element::Type& get_element_type() const override;
    const ov::Shape& get_shape() const override;
    const ov::Strides& get_strides() const override;

    void allocate();
    bool deallocate() noexcept;

    bool is_allocated() const noexcept;
    bool is_surface() const noexcept;
    bool is_shared() const noexcept;
    cldnn::memory::ptr get_memory() const;
    cldnn::memory::ptr get_original_memory() const;

    void set_memory(cldnn::memory::ptr memory, size_t actual_size);

    std::shared_ptr<TupleRemoteContextImpl> get_context() const;
    ov::SoPtr<ov::IRemoteTensor> get_tensor(int index) const;

private:
    std::shared_ptr<TupleRemoteContextImpl> m_context;

    ov::element::Type m_element_type;
    ov::Shape m_shape;
    // std::vector<std::shared_ptr<RemoteContextImpl>> m_contexts;

    // std::vector<ov::element::Type> m_element_types;

    std::vector<ov::Shape> m_shapes;
    ov::Strides m_strides{};
    ov::AnyMap m_properties;

    cldnn::memory::ptr m_memory_object = nullptr;
    cldnn::layout m_layout;
    TensorType m_mem_type;

    cldnn::shared_handle m_mem;
    cldnn::shared_surface m_surf;
    uint32_t m_plane;
    size_t m_hash = 0;
    std::vector<ov::SoPtr<ov::IRemoteTensor>> m_tensors;

    bool supports_caching() const;
    void update_hash();
    void update_strides();
    void update_properties();

    static TensorType allocation_type_to_tensor_type(cldnn::allocation_type t);
};

}  // namespace intel_gpu
}  // namespace ov
