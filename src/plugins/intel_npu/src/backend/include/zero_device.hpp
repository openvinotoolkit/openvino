// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include "intel_npu/common/icompiled_model.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"

namespace ov::op::v0 {
class Constant;
}

namespace intel_npu {

class ZeroDevice : public IDevice {
public:
    ZeroDevice(const std::shared_ptr<ZeroInitStructsHolder>& initStructs);

    /**
     * @brief TODO
     */
    std::pair<std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>, ov::SoPtr<ov::ITensor>> runInit(
        const std::shared_ptr<IGraph>& initGraph,
        const std::shared_ptr<const ov::Model>& model,
        const ov::SoPtr<ov::IRemoteContext>& context,
        const Config& config) override;

    std::pair<std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>, std::vector<ov::SoPtr<ov::ITensor>>>
    runInitMultiThreaded(const std::vector<std::shared_ptr<IGraph>>& initGraph,
                         const std::shared_ptr<const ov::Model>& model,
                         const ov::SoPtr<ov::IRemoteContext>& context,
                         const Config& config) override;

    std::string getName() const override;
    std::string getFullDeviceName() const override;
    Uuid getUuid() const override;
    ov::device::LUID getLUID() const override;
    uint32_t getSubDevId() const override;
    uint32_t getMaxNumSlices() const override;
    uint64_t getAllocMemSize() const override;
    uint64_t getTotalMemSize() const override;
    ov::device::PCIInfo getPciInfo() const override;
    std::map<ov::element::Type, float> getGops() const override;
    ov::device::Type getDeviceType() const override;

    std::shared_ptr<SyncInferRequest> createInferRequest(const std::shared_ptr<const ICompiledModel>& compiledModel,
                                                         const Config& config) override;
    void updateInfo(const Config& config) override {
        log.setLevel(config.get<LOG_LEVEL>());
    }

<<<<<<< HEAD
    ov::SoPtr<ov::IRemoteTensor> createRemoteTensor(
        std::shared_ptr<ov::IRemoteContext> context,
        const ov::element::Type& element_type,
        const ov::Shape& shape,
        const Config& config,
        ov::intel_npu::TensorType tensor_type = ov::intel_npu::TensorType::BINDED,
        ov::intel_npu::MemType mem_type = ov::intel_npu::MemType::L0_INTERNAL_BUF,
        const void* mem = nullptr) override;

    ov::SoPtr<ov::ITensor> createHostTensor(
        std::shared_ptr<ov::IRemoteContext> context,
        const ov::element::Type& element_type,
        const ov::Shape& shape,
        const Config& config,
        ov::intel_npu::TensorType tensor_type = ov::intel_npu::TensorType::BINDED) override;

=======
>>>>>>> upstream/master
    ZeroDevice& operator=(const ZeroDevice&) = delete;
    ZeroDevice(const ZeroDevice&) = delete;

    // TODO: public for multi-threaded execution
    struct InputData {
        // TODO: is it necessary to keep both fields alive? it doesn't seem like
        // hostTensor field is ever used.
        std::vector<std::vector<std::shared_ptr<ov::ITensor>>> tensors;
        ov::SoPtr<ov::ITensor> hostTensor;
    };

    struct OutputData {
        // TODO: is it necessary to keep both fields alive? it doesn't seem like
        // hostTensor field is ever used.
        std::vector<std::shared_ptr<ov::ITensor>> tensors;
        ov::SoPtr<ov::ITensor> hostTensor;
        std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> tensorsMap;
    };

private:
    InputData allocateInputs(const std::shared_ptr<IGraph>& initGraph,
                             const std::vector<std::shared_ptr<ov::op::v0::Constant>>& constants,
                             const ov::SoPtr<ov::IRemoteContext>& context,
                             const Config& config);

    OutputData allocateOutputs(const std::shared_ptr<IGraph>& initGraph,
                               const ov::SoPtr<ov::IRemoteContext>& context,
                               const Config& config);

    const std::shared_ptr<ZeroInitStructsHolder> _initStructs;

    ze_device_properties_t device_properties = {};

    ze_pci_ext_properties_t pci_properties = {};

    ze_device_luid_ext_properties_t device_luid = {};

    std::map<ov::element::Type, float> device_gops = {{ov::element::f32, 0.f},
                                                      {ov::element::f16, 0.f},
                                                      {ov::element::bf16, 0.f},
                                                      {ov::element::u8, 0.f},
                                                      {ov::element::i8, 0.f}};

    Logger log;
};
}  // namespace intel_npu
