// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime IRemoteTensor interface
 * @file openvino/runtime/iremote_tensor.hpp
 */

#pragma once

#include "openvino/core/except.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/itensor.hpp"

namespace ov {

class OPENVINO_RUNTIME_API IRemoteTensor : public ITensor {
public:
    void* data(const element::Type& type = {}) const override final {
        OPENVINO_NOT_IMPLEMENTED;
    }

    ~IRemoteTensor() override;

    /**
     * @brief Returns additional information associated with tensor
     * @return Map of property names to properties
     */
    virtual const AnyMap& get_properties() const = 0;
    /**
     * @brief Returns device name
     * @return Device name
     */
    virtual const std::string& get_device_name() const = 0;
};
}  // namespace ov
