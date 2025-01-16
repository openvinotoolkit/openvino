// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "intel_npu/config/config.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/iremote_tensor.hpp"

namespace intel_npu {

/**
 * @brief Acts as an interface for the remote tensor structures implemented by all backends.
 * @details The operations common for all backends can be found implemented here
 */
class RemoteTensor : public ov::IRemoteTensor {
public:
    RemoteTensor(const std::shared_ptr<ov::IRemoteContext>& context,
                 const ov::element::Type& element_type,
                 const ov::Shape& shape);

    /**
     * @brief Returns additional information associated with tensor
     * @return Map of property names to properties
     */
    const ov::AnyMap& get_properties() const override;

    /**
     * @brief Returns device name
     * @return Device name
     */
    const std::string& get_device_name() const override;

    /**
     * @brief Set new shape for tensor
     * @note Allocation of a bigger tensor is not possible
     * @param shape A new shape
     */
    void set_shape(ov::Shape shape) override;

    /**
     * @return A tensor element type
     */
    const ov::element::Type& get_element_type() const override;

    /**
     * @return A tensor shape
     */
    const ov::Shape& get_shape() const override;

    /**
     * @return Tensor's strides in bytes
     */
    const ov::Strides& get_strides() const override;

    /**
     * @return The remote context
     */
    std::shared_ptr<ov::IRemoteContext> get_context() const;

protected:
    virtual void allocate(const size_t bytes) = 0;
    virtual bool deallocate() noexcept = 0;
    void update_strides();

    virtual ~RemoteTensor();

    std::shared_ptr<ov::IRemoteContext> _context;

    ov::element::Type _element_type;
    ov::Shape _shape;
    ov::Shape _capacity;
    ov::Strides _strides{};
    ov::AnyMap _properties;
};

}  // namespace intel_npu
