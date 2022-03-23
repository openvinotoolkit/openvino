// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the IRemoteContext
 *
 * @file iremote_context.hpp
 */
#pragma once

#include "ie_remote_context.hpp"
#include "openvino/runtime/common.hpp"
namespace ov {

class ITensor;

/**
 * @brief This class represents an OpenVINO abstraction
 * for remote (non-CPU) accelerator device-specific execution context.
 * Such context represents a scope on the device within which executable
 * networks and remote memory blobs can exist, function and exchange data.
 */
class OPENVINO_API IRemoteContext : public std::enable_shared_from_this<IRemoteContext> {
public:
    /**
     * @brief A smart pointer containing IRemoteContext object
     */
    using Ptr = std::shared_ptr<IRemoteContext>;

    /**
     * @brief Returns name of a device on which underlying object is allocated.
     * Abstract method.
     * @return A device name string in fully specified format `<device_name>[.<device_id>[.<tile_id>]]` (e.g. GPU.0.1).
     */
    virtual std::string get_device_name() const;

    /**
     * @brief Allocates memory tensor in device memory or wraps user-supplied memory handle
     * using the specified tensor description and low-level device-specific parameters.
     * Returns a pointer to the object that implements the RemoteTensor interface.
     * @param type Defines the element type of the tensor.
     * @param shape Defines the shape of the tensor.
     * @param params Map of the low-level tensor object parameters.
     * @return Pointer to a plugin object that implements the RemoteTensor interface.
     */
    virtual std::shared_ptr<ITensor> create_tensor(const element::Type type,
                                                   const Shape& shape,
                                                   const AnyMap& params = {});

    /**
     * @brief Returns a map of device-specific parameters required for low-level
     * operations with the underlying object.
     * Parameters include device/context handles, access flags,
     * etc. Content of the returned map depends on a remote execution context that is
     * currently set on the device (working scenario).
     * Abstract method.
     * @return A map of name/parameter elements.
     */
    virtual AnyMap get_params() const;

    /**
     * @brief This method is used to create a host tensor object friendly for the device in current context.
     * For example, GPU context may allocate USM host memory (if corresponding extension is available),
     * which could be more efficient than regular host memory.
     * @param type Tensor element type.
     * @param shape Tensor shape.
     * @return A tensor instance with device friendly memory.
     */
    virtual std::shared_ptr<ITensor> create_host_tensor(const element::Type type, const Shape& shape);

protected:
    /**
     * @brief RemoteContext destructor
     */
    ~IRemoteContext() = default;
};

struct OPENVINO_API IERemoteContext : public IRemoteContext {
    explicit IERemoteContext(const InferenceEngine::RemoteContext::Ptr& impl_);
    std::string get_device_name() const override;
    std::shared_ptr<ITensor> create_tensor(const element::Type type, const Shape& shape, const AnyMap& params) override;
    AnyMap get_params() const override;
    std::shared_ptr<ITensor> create_host_tensor(const element::Type type, const Shape& shape) override;

    InferenceEngine::RemoteContext::Ptr impl;
};

}  // namespace ov

namespace InferenceEngine {
struct OPENVINO_API OVRemoteContext : public RemoteContext {
    OVRemoteContext(const ov::IRemoteContext::Ptr& impl_) : impl{impl_} {}
    ~OVRemoteContext() override = default;
    std::string getDeviceName() const noexcept override;
    RemoteBlob::Ptr CreateBlob(const TensorDesc& tensorDesc, const ParamMap& params = {}) override;
    MemoryBlob::Ptr CreateHostBlob(const TensorDesc& tensorDesc) override;
    ParamMap getParams() const override;
    ov::IRemoteContext::Ptr impl;
};
}  // namespace InferenceEngine