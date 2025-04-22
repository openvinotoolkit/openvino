// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "intel_npu/common/npu.hpp"
#include "intel_npu/config/config.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "openvino/runtime/iremote_context.hpp"

namespace intel_npu {

class RemoteContextImpl : public ov::IRemoteContext {
public:
    RemoteContextImpl(const ov::SoPtr<IEngineBackend>& engineBackend,
                      const Config& config,
                      const ov::AnyMap& remote_properties = {});

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

    const Config _config;
    std::shared_ptr<intel_npu::IDevice> _device;
    ov::AnyMap _properties;
    std::string _device_name;

    std::optional<ov::intel_npu::MemType> _mem_type_object = std::nullopt;
    std::optional<ov::intel_npu::TensorType> _tensor_type_object = std::nullopt;
    std::optional<void*> _mem_handle_object = std::nullopt;
};

}  // namespace intel_npu
