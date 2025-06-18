// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/runtime/itensor.hpp"
#include "zero_remote_tensor.hpp"

namespace intel_npu {

class ZeroHostTensor : public ov::ITensor {
public:
    ZeroHostTensor(const std::shared_ptr<ov::IRemoteContext>& context,
                   const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                   const ov::element::Type element_type,
                   const ov::Shape& shape,
                   ov::intel_npu::TensorType tensor_type = ov::intel_npu::TensorType::BINDED);

    ~ZeroHostTensor() override = default;

    void* data() override;
    void* data(const ov::element::Type& element_type) override;
    const void* data() const override;
    const void* data(const ov::element::Type& element_type) const override;
    const ov::element::Type& get_element_type() const override;

    const ov::Shape& get_shape() const override;

    const ov::Strides& get_strides() const override;

    void set_shape(ov::Shape new_shape) override;

    std::shared_ptr<ZeroRemoteTensor> get_impl() const;

private:
    std::shared_ptr<ZeroRemoteTensor> _impl;
};

}  // namespace intel_npu
