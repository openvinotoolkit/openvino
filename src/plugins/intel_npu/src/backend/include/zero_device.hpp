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

namespace intel_npu {

class ZeroDevice : public IDevice {
public:
    ZeroDevice(const std::shared_ptr<ZeroInitStructsHolder>& initStructs);

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

    std::shared_ptr<ov::IInferRequest> createInferRequest(const std::shared_ptr<const ICompiledModel>& compiledModel,
                                                          const Config& config,
                                                          std::function<void(void)>& inferAsyncF,
                                                          std::function<void(void)>& getResultF) override;
    void updateInfo(const Config& config) override {
        log.setLevel(config.get<LOG_LEVEL>());
    }

    ZeroDevice& operator=(const ZeroDevice&) = delete;
    ZeroDevice(const ZeroDevice&) = delete;

    ~ZeroDevice() = default;

private:
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
