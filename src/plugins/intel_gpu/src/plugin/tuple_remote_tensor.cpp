// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/tuple_remote_tensor.hpp"

#include <memory>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/plugin.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
namespace ov {
namespace intel_gpu {

TupleRemoteTensorImpl::TupleRemoteTensorImpl(std::shared_ptr<TupleRemoteContextImpl> context,
                                             std::vector<ov::SoPtr<ov::IRemoteTensor>> tensors)
    : m_context(context),
      m_ordered_tensor(tensors) {
    for (auto& tensor : tensors) {
        auto remote_tensor = std::dynamic_pointer_cast<RemoteTensorImpl>(tensor._ptr);
        m_remote_tensors.emplace_back(remote_tensor);
        auto device_name = remote_tensor->get_device_name();
        m_tensors.insert({device_name, tensor});
    }
}

TupleRemoteTensorImpl::~TupleRemoteTensorImpl() {
    deallocate();
}

ov::SoPtr<ov::IRemoteTensor> TupleRemoteTensorImpl::get_tensor(int index) const {
    return m_ordered_tensor[index];
}

ov::SoPtr<ov::IRemoteTensor> TupleRemoteTensorImpl::get_tensor_by_name(const std::string device_name) const {
    return m_tensors.at(device_name);
}

const ov::element::Type& TupleRemoteTensorImpl::get_element_type() const {
    return m_tensors.begin()->second->get_element_type();
}

const ov::Shape& TupleRemoteTensorImpl::get_shape() const {
    return m_tensors.begin()->second->get_shape();
}

const ov::Strides& TupleRemoteTensorImpl::get_strides() const {
    return m_tensors.begin()->second->get_strides();
}

const AnyMap& TupleRemoteTensorImpl::get_properties() const {
    return m_tensors.begin()->second->get_properties();
}

void TupleRemoteTensorImpl::set_shape(ov::Shape shape) {
    for (auto it = m_tensors.begin(); it != m_tensors.end(); ++it) {
        it->second->set_shape(shape);
    }
}

bool TupleRemoteTensorImpl::deallocate() noexcept {
    bool deallocate = true;
    for (auto& tensor : m_remote_tensors) {
        deallocate &= tensor->deallocate();
    }
    return deallocate;
}

bool TupleRemoteTensorImpl::is_allocated() const noexcept {
    bool is_allocated = true;
    for (auto& tensor : m_remote_tensors) {
        is_allocated &= tensor->is_allocated();
    }
    return is_allocated;
}

void TupleRemoteTensorImpl::allocate() {
    for (auto& tensor : m_remote_tensors) {
        tensor->allocate();
    }
}

const std::string& TupleRemoteTensorImpl::get_device_name() const {
    return m_context->get_device_name();
}

void TupleRemoteTensorImpl::set_memory(cldnn::memory::ptr memory, size_t actual_size) {
    for (auto& tensor : m_remote_tensors) {
        tensor->set_memory(memory, actual_size);
    }
}

std::shared_ptr<TupleRemoteContextImpl> TupleRemoteTensorImpl::get_context() const {
    return m_context;
}

void TupleRemoteTensorImpl::copy_to(const std::shared_ptr<ov::ITensor>& dst, size_t src_offset, size_t dst_offset, const ov::Shape& roi_shape) const {
    if (auto remote = std::dynamic_pointer_cast<ov::intel_gpu::TupleRemoteTensorImpl>(dst)) {
        int i = 0;
        for (auto& tensor : m_remote_tensors) {
            auto itensor = std::dynamic_pointer_cast<ov::ITensor>(remote->get_tensor(i)._ptr);
            tensor->copy_to(itensor, src_offset, dst_offset, roi_shape);
            i++;
        }
    } else {
        int i = 0;
        for (auto& tensor : m_remote_tensors) {
            tensor->copy_to(dst, src_offset, dst_offset + i * get_strides()[0], roi_shape);
            i++;
        }
    }
}

void TupleRemoteTensorImpl::copy_from(const std::shared_ptr<const ov::ITensor>& src, size_t src_offset, size_t dst_offset, const ov::Shape& roi_shape) {
    if (auto remote = std::dynamic_pointer_cast<const ov::intel_gpu::TupleRemoteTensorImpl>(src)) {
        int i = 0;
        for (auto& tensor : m_remote_tensors) {
            auto itensor = std::dynamic_pointer_cast<ov::ITensor>(remote->get_tensor(i)._ptr);
            tensor->copy_from(itensor, src_offset, dst_offset, roi_shape);
            i++;
        }
    } else {
        auto new_roi_shape = get_shape();
        new_roi_shape[0] = roi_shape[0];
        int i = 0;
        for (auto& tensor : m_remote_tensors) {
            tensor->copy_from(src, src_offset + i * get_strides()[0], dst_offset, new_roi_shape);
            i++;
        }
    }
}

}  // namespace intel_gpu
}  // namespace ov
