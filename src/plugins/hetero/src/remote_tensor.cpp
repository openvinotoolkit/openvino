// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_tensor.hpp"

namespace ov {
namespace hetero {

RemoteTensor::RemoteTensor(const std::shared_ptr<RemoteContext>& context,
                           std::vector<ov::SoPtr<ov::IRemoteTensor>> tensors)
    : m_context(context),
      m_ordered_tensor(std::move(tensors)) {
    m_remote_tensors.reserve(m_ordered_tensor.size());
    for (const auto& tensor : m_ordered_tensor) {
        auto remote_tensor = std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr);
        OPENVINO_ASSERT(remote_tensor, "Invalid tensor pointer in RemoteTensor constructor");
        m_remote_tensors.emplace_back(remote_tensor);

        const auto& device_name = remote_tensor->get_device_name();
        m_tensors.insert({device_name, tensor});
    }
}

const std::string& RemoteTensor::get_device_name() const {
    return m_context->get_device_name();
}

const ov::element::Type& RemoteTensor::get_element_type() const {
    return m_tensors.begin()->second->get_element_type();
}

const ov::Strides& RemoteTensor::get_strides() const {
    return m_tensors.begin()->second->get_strides();
}

const AnyMap& RemoteTensor::get_properties() const {
    return m_tensors.begin()->second->get_properties();
}

const ov::Shape& RemoteTensor::get_shape() const {
    return m_tensors.begin()->second->get_shape();
}

std::shared_ptr<RemoteContext> RemoteTensor::get_context() const {
    return m_context;
}

ov::SoPtr<ov::IRemoteTensor> RemoteTensor::get_tensor(int index) const {
    return m_ordered_tensor[index];
}

ov::SoPtr<ov::IRemoteTensor> RemoteTensor::get_tensor_by_name(const std::string device_name) const {
    return m_tensors.at(device_name);
}

void RemoteTensor::set_shape(ov::Shape shape) {
    for (auto it = m_tensors.begin(); it != m_tensors.end(); ++it) {
        it->second->set_shape(shape);
    }
}

void RemoteTensor::copy_to(const std::shared_ptr<ov::ITensor>& dst,
                           size_t src_offset,
                           size_t dst_offset,
                           const ov::Shape& roi_shape) const {
    if (auto remote = std::dynamic_pointer_cast<ov::hetero::RemoteTensor>(dst)) {
        for (size_t i = 0; i < m_remote_tensors.size(); ++i) {
            auto itensor = std::dynamic_pointer_cast<ov::ITensor>(remote->get_tensor(static_cast<int>(i))._ptr);
            m_remote_tensors[i]->copy_to(itensor, src_offset, dst_offset, roi_shape);
        }
    } else {
        int i = 0;
        for (auto& tensor : m_remote_tensors) {
            tensor->copy_to(dst, src_offset, dst_offset + i * get_strides()[0], roi_shape);
            i++;
        }
    }
}

void RemoteTensor::copy_from(const std::shared_ptr<const ov::ITensor>& src,
                             size_t src_offset,
                             size_t dst_offset,
                             const ov::Shape& roi_shape) {
    if (auto remote = std::dynamic_pointer_cast<const ov::hetero::RemoteTensor>(src)) {
        for (size_t i = 0; i < m_remote_tensors.size(); ++i) {
            auto itensor = std::dynamic_pointer_cast<ov::ITensor>(remote->get_tensor(static_cast<int>(i))._ptr);
            m_remote_tensors[i]->copy_from(itensor, src_offset, dst_offset, roi_shape);
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
