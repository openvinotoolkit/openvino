// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "backends.hpp"
#include "intel_npu/config/config.hpp"
#include "openvino/runtime/iremote_context.hpp"

namespace intel_npu {

class RemoteContextImpl : public ov::IRemoteContext {
public:
    RemoteContextImpl(std::shared_ptr<const NPUBackends> backends, const Config& config);

    /**
     * @brief Returns name of a device on which underlying object is allocated.
     * @return A device name string in fully specified format `<device_name>[.<device_id>[.<tile_id>]]` (e.g. GPU.0.1).
     */
    const std::string& get_device_name() const override;

    /**
     * @brief Returns a map of device-specific parameters
     * @return A map of name/Any elements.
     */
    const ov::AnyMap& get_property() const override;

    /**
     * @brief Allocates memory tensor in device memory or wraps user-supplied memory handle
     * using the specified tensor description and low-level device-specific parameters.
     * Returns a pointer to the object that implements the RemoteTensor interface.
     * @param type Defines the element type of the tensor.
     * @param shape Defines the shape of the tensor.
     * @param params Map of the low-level tensor object parameters.
     * @return Pointer to a plugin object that implements the RemoteTensor interface.
     */
    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type,
                                               const ov::Shape& shape,
                                               const ov::AnyMap& params) override;

    /**
     * @brief This method is used to create a host tensor object friendly for the device in current context.
     * @param type Tensor element type.
     * @param shape Tensor shape.
     * @return A tensor instance with device friendly memory.
     */
    ov::SoPtr<ov::ITensor> create_host_tensor(const ov::element::Type type, const ov::Shape& shape) override;

private:
    std::shared_ptr<ov::IRemoteContext> get_this_shared_ptr();

    std::shared_ptr<const NPUBackends> _backends;

    const Config _config;
    ov::AnyMap _properties;
    std::string _device_name;
};

}  // namespace intel_npu
