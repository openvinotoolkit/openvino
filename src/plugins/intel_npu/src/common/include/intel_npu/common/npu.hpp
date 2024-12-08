// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "intel_npu/common/icompiled_model.hpp"
#include "intel_npu/common/igraph.hpp"
#include "intel_npu/common/sync_infer_request.hpp"
#include "intel_npu/config/config.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/properties.hpp"

namespace intel_npu {

//------------------------------------------------------------------------------
class IDevice;

class IEngineBackend : public std::enable_shared_from_this<IEngineBackend> {
public:
    /** @brief Get device, which can be used for inference. Backend responsible for selection. */
    virtual const std::shared_ptr<IDevice> getDevice() const;
    /** @brief Search for a specific device by name */
    virtual const std::shared_ptr<IDevice> getDevice(const std::string& specificDeviceName) const;
    /** @brief Get device, which is configured/suitable for provided params */
    virtual const std::shared_ptr<IDevice> getDevice(const ov::AnyMap& paramMap) const;
    /** @brief Provide a list of names of all devices, with which user can work directly */
    virtual const std::vector<std::string> getDeviceNames() const;
    /** @brief Provide driver version */
    virtual uint32_t getDriverVersion() const;
    /** @brief Provide driver extension version */
    virtual uint32_t getGraphExtVersion() const;
    /** @brief Get name of backend */
    virtual const std::string getName() const = 0;
    /** @brief Backend has support for concurrency batching */
    virtual bool isBatchingSupported() const = 0;
    /** @brief Backend has support for workload type */
    virtual bool isCommandQueueExtSupported() const = 0;
    /** @brief Backend has support for LUID info */
    virtual bool isLUIDExtSupported() const = 0;
    /** @brief Register backend-specific options */
    virtual void registerOptions(OptionsDesc& options) const;
    /** @brief Get Level Zero context*/
    virtual void* getContext() const;
    /** @brief Update backend and device info */
    virtual void updateInfo(const Config& config) = 0;

protected:
    virtual ~IEngineBackend() = default;
};

//------------------------------------------------------------------------------

class ICompilerAdapter {
public:
    virtual std::shared_ptr<IGraph> compile(const std::shared_ptr<const ov::Model>& model,
                                            const Config& config) const = 0;
    virtual std::shared_ptr<IGraph> parse(std::vector<uint8_t> network, const Config& config) const = 0;
    virtual ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const = 0;

    virtual ~ICompilerAdapter() = default;
};

//------------------------------------------------------------------------------

class IDevice : public std::enable_shared_from_this<IDevice> {
public:
    using Uuid = ov::device::UUID;

    virtual std::string getName() const = 0;
    virtual std::string getFullDeviceName() const = 0;
    virtual Uuid getUuid() const;
    virtual ov::device::LUID getLUID() const;
    virtual uint32_t getSubDevId() const;
    virtual uint32_t getMaxNumSlices() const;
    virtual uint64_t getAllocMemSize() const;
    virtual uint64_t getTotalMemSize() const;
    virtual ov::device::PCIInfo getPciInfo() const;
    virtual ov::device::Type getDeviceType() const;
    virtual std::map<ov::element::Type, float> getGops() const;

    virtual std::shared_ptr<SyncInferRequest> createInferRequest(
        const std::shared_ptr<const ICompiledModel>& compiledModel,
        const Config& config) = 0;

    virtual void updateInfo(const Config& config) = 0;

    virtual ov::SoPtr<ov::IRemoteTensor> createRemoteTensor(
        std::shared_ptr<ov::IRemoteContext> context,
        const ov::element::Type& element_type,
        const ov::Shape& shape,
        const Config& config,
        ov::intel_npu::TensorType tensor_type = ov::intel_npu::TensorType::BINDED,
        ov::intel_npu::MemType mem_type = ov::intel_npu::MemType::L0_INTERNAL_BUF,
        void* mem = nullptr);

    virtual ov::SoPtr<ov::ITensor> createHostTensor(std::shared_ptr<ov::IRemoteContext> context,
                                                    const ov::element::Type& element_type,
                                                    const ov::Shape& shape,
                                                    const Config& config);

protected:
    virtual ~IDevice() = default;
};

}  // namespace intel_npu
