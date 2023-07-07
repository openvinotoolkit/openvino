// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/remote_tensor.hpp"

namespace ov {
namespace proxy {

/**
 * @brief Proxy remote tensor class.
 * This class wraps the original remote tensor and change the name of RemoteTensor
 */
class RemoteTensor : public ov::IRemoteTensor {
public:
    RemoteTensor(ov::RemoteTensor&& ctx, const std::string& dev_name);
    RemoteTensor(const ov::RemoteTensor& ctx, const std::string& dev_name);

    const AnyMap& get_properties() const override;
    const std::string& get_device_name() const override;

    void set_shape(ov::Shape shape) override;

    const ov::element::Type& get_element_type() const override;

    const ov::Shape& get_shape() const override;

    size_t get_size() const override;

    size_t get_byte_size() const override;

    const ov::Strides& get_strides() const override;

    static const std::shared_ptr<ov::ITensor>& get_hardware_tensor(const std::shared_ptr<ov::ITensor>& tensor);

private:
    mutable std::string m_name;
    ov::RemoteTensor m_tensor;
};

}  // namespace proxy
}  // namespace ov
