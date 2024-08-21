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

    void copy_to(const std::shared_ptr<ov::ITensor>& dst) const override {
        const auto zero_offset = 0;
        copy_to(dst, zero_offset, zero_offset, {});
    }

    virtual void copy_from(const std::shared_ptr<const ov::ITensor>& src) {
        const auto zero_offset = 0;
        copy_from(src, zero_offset, zero_offset, {});
    }

    virtual void copy_to(const std::shared_ptr<ov::ITensor>& dst, size_t src_offset, size_t dst_offset, const ov::Shape& roi_shape) const {
        OPENVINO_NOT_IMPLEMENTED;
    };

    virtual void copy_from(const std::shared_ptr<const ov::ITensor>& src, size_t src_offset, size_t dst_offset, const ov::Shape& roi_shape) {
        OPENVINO_NOT_IMPLEMENTED;
    };
};
}  // namespace ov
