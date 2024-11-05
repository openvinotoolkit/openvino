// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/config/config.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/runtime/itensor.hpp"
#include "zero_remote_tensor.hpp"

namespace intel_npu {

class ZeroHostTensor : public ov::ITensor {
public:
    ZeroHostTensor(std::shared_ptr<ov::IRemoteContext> context,
                   std::shared_ptr<ZeroInitStructsHolder> init_structs,
                   const ov::element::Type element_type,
                   const ov::Shape& shape,
                   const Config& config);

    ~ZeroHostTensor() override = default;

    void* data(const ov::element::Type& element_type) const override;
    const ov::element::Type& get_element_type() const override;

    const ov::Shape& get_shape() const override;

    const ov::Strides& get_strides() const override;

    void set_shape(ov::Shape new_shape) override;

    std::shared_ptr<ZeroRemoteTensor> get_impl() const;

private:
    std::shared_ptr<ZeroRemoteTensor> m_impl;
};

}  // namespace intel_npu
