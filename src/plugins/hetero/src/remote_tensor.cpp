// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_tensor.hpp"

namespace ov {
namespace hetero {

HeteroRemoteTensor::HeteroRemoteTensor(std::shared_ptr<HeteroContext> context,
                                       std::vector<ov::SoPtr<ov::IRemoteTensor>> tensors)
    : m_context(context),
      m_ordered_tensor(tensors) {
    for (auto& tensor : tensors) {
        auto remote_tensor = std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr);
        m_remote_tensors.emplace_back(remote_tensor);
        auto device_name = remote_tensor->get_device_name();
        m_tensors.insert({device_name, tensor});
    }
}

const std::string& HeteroRemoteTensor::get_device_name() const {
    return m_context->get_device_name();
}

const ov::element::Type& HeteroRemoteTensor::get_element_type() const {
    return m_tensors.begin()->second->get_element_type();
}

const ov::Strides& HeteroRemoteTensor::get_strides() const {
    return m_tensors.begin()->second->get_strides();
}

const AnyMap& HeteroRemoteTensor::get_properties() const {
    return m_tensors.begin()->second->get_properties();
}

const ov::Shape& HeteroRemoteTensor::get_shape() const {
    return m_tensors.begin()->second->get_shape();
}

std::shared_ptr<HeteroContext> HeteroRemoteTensor::get_context() const {
    return m_context;
}

ov::SoPtr<ov::IRemoteTensor> HeteroRemoteTensor::get_tensor(int index) const {
    return m_ordered_tensor[index];
}

ov::SoPtr<ov::IRemoteTensor> HeteroRemoteTensor::get_tensor_by_name(const std::string device_name) const {
    return m_tensors.at(device_name);
}

void HeteroRemoteTensor::set_shape(ov::Shape shape) {
    for (auto it = m_tensors.begin(); it != m_tensors.end(); ++it) {
        it->second->set_shape(shape);
    }
}

void HeteroRemoteTensor::copy_to(const std::shared_ptr<ov::ITensor>& dst,
                                 size_t src_offset,
                                 size_t dst_offset,
                                 const ov::Shape& roi_shape) const {
    if (auto remote = std::dynamic_pointer_cast<ov::hetero::HeteroRemoteTensor>(dst)) {
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

void HeteroRemoteTensor::copy_from(const std::shared_ptr<const ov::ITensor>& src,
                                   size_t src_offset,
                                   size_t dst_offset,
                                   const ov::Shape& roi_shape) {
    if (auto remote = std::dynamic_pointer_cast<const ov::hetero::HeteroRemoteTensor>(src)) {
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

}  // namespace hetero
}  // namespace ov
