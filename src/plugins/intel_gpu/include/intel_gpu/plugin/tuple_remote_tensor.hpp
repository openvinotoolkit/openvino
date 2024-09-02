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

    void set_memory(cldnn::memory::ptr memory, size_t actual_size);

    std::shared_ptr<TupleRemoteContextImpl> get_context() const;
    ov::SoPtr<ov::IRemoteTensor> get_tensor(int index) const;
    ov::SoPtr<ov::IRemoteTensor> get_tensor_by_name(const std::string device_name) const;

    void copy_to(const std::shared_ptr<ov::ITensor>& dst, size_t src_offset, size_t dst_offset, const ov::Shape& roi_shape) const override;
    void copy_from(const std::shared_ptr<const ov::ITensor>& src, size_t src_offset, size_t dst_offset, const ov::Shape& roi_shape) override;

private:
    std::shared_ptr<TupleRemoteContextImpl> m_context;
    std::vector<ov::SoPtr<ov::IRemoteTensor>> m_ordered_tensor;
    std::map<std::string, ov::SoPtr<ov::IRemoteTensor>> m_tensors;
    std::vector<std::shared_ptr<RemoteTensorImpl>> m_remote_tensors;
};

}  // namespace intel_gpu
}  // namespace ov
