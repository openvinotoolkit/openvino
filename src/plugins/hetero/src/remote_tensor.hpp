// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "context.hpp"
#include "openvino/runtime/iremote_tensor.hpp"

namespace ov {
namespace hetero {

class HeteroRemoteTensor : public ov::IRemoteTensor {
public:
    HeteroRemoteTensor(std::shared_ptr<ov::hetero::HeteroContext> context,
                       std::vector<ov::SoPtr<ov::IRemoteTensor>> tensors);

    const std::string& get_device_name() const override;

    const ov::element::Type& get_element_type() const override;

    const ov::Strides& get_strides() const override;

    const AnyMap& get_properties() const override;

    const ov::Shape& get_shape() const override;

    std::shared_ptr<HeteroContext> get_context() const;

    ov::SoPtr<ov::IRemoteTensor> get_tensor(int index) const;

    ov::SoPtr<ov::IRemoteTensor> get_tensor_by_name(const std::string device_name) const;

    void set_shape(ov::Shape shape) override;

    void copy_to(const std::shared_ptr<ov::ITensor>& dst,
                 size_t src_offset,
                 size_t dst_offset,
                 const ov::Shape& roi_shape) const override;

    void copy_from(const std::shared_ptr<const ov::ITensor>& src,
                   size_t src_offset,
                   size_t dst_offset,
                   const ov::Shape& roi_shape) override;

private:
    std::shared_ptr<HeteroContext> m_context;
    std::vector<ov::SoPtr<ov::IRemoteTensor>> m_ordered_tensor;
    std::map<std::string, ov::SoPtr<ov::IRemoteTensor>> m_tensors;
    std::vector<std::shared_ptr<ov::IRemoteTensor>> m_remote_tensors;
};

}  // namespace hetero
}  // namespace ov
