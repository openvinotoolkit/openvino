//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "npu/utils/logger/logger.hpp"
#include "vpux.hpp"
#include "vpux/al/icompiled_model.hpp"
#include "ze_api.h"
#include "ze_graph_ext.h"
#include "zero_init.h"
#include "zero_types.h"
#include "zero_utils.h"

using intel_npu::Logger;

namespace vpux {

class ZeroDevice : public IDevice {
public:
    ZeroDevice(const std::shared_ptr<ZeroInitStructsHolder>& initStructs);

    std::shared_ptr<IExecutor> createExecutor(const std::shared_ptr<const NetworkDescription>& networkDescription,
                                              const Config& config) override;

    std::string getName() const override;
    std::string getFullDeviceName() const override;
    Uuid getUuid() const override;
    uint32_t getSubDevId() const override;
    uint32_t getMaxNumSlices() const override;
    uint64_t getAllocMemSize() const override;
    uint64_t getTotalMemSize() const override;
    uint32_t getDriverVersion() const override;

    std::shared_ptr<SyncInferRequest> createInferRequest(
        const std::shared_ptr<const vpux::ICompiledModel>& compiledModel,
        const std::shared_ptr<IExecutor>& executor,
        const Config& config) override;

    ZeroDevice& operator=(const ZeroDevice&) = delete;
    ZeroDevice(const ZeroDevice&) = delete;

private:
    const std::shared_ptr<ZeroInitStructsHolder> _initStructs;

    ze_graph_dditable_ext_curr_t* _graph_ddi_table_ext = nullptr;

    ze_device_properties_t device_properties = {};
    ze_driver_properties_t driver_properties = {};

    uint32_t _group_ordinal;

    Logger log;
};
}  // namespace vpux
