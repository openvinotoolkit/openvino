// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/itensor.hpp"

#include <memory>

namespace ov::intel_gpu {

class RemoteContextImpl;
class RemoteTensorImpl;

class USMHostTensor : public ov::ITensor {
public:
    USMHostTensor(std::shared_ptr<RemoteContextImpl> context, const element::Type element_type, const Shape& shape);
    explicit USMHostTensor(std::shared_ptr<RemoteTensorImpl> tensor);

    ~USMHostTensor() override = default;

    void* data(const element::Type& element_type) const override;
    const element::Type& get_element_type() const override;

    const Shape& get_shape() const override;

    const Strides& get_strides() const override;

    void set_shape(ov::Shape new_shape) override;

    void set_memory(std::shared_ptr<RemoteTensorImpl> tensor);

    std::shared_ptr<RemoteTensorImpl> get_impl() const;

private:
    std::shared_ptr<RemoteTensorImpl> m_impl;
};

}  // namespace ov::intel_gpu
