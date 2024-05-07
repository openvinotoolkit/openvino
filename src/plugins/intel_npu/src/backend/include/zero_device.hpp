// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include "intel_npu/al/icompiled_model.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "npu.hpp"
#include "zero_init.hpp"
#include "zero_types.hpp"
#include "zero_utils.hpp"
namespace intel_npu {

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

    std::shared_ptr<SyncInferRequest> createInferRequest(const std::shared_ptr<const ICompiledModel>& compiledModel,
                                                         const std::shared_ptr<IExecutor>& executor,
                                                         const Config& config) override;

    ZeroDevice& operator=(const ZeroDevice&) = delete;
    ZeroDevice(const ZeroDevice&) = delete;

private:
    const std::shared_ptr<ZeroInitStructsHolder> _initStructs;

    ze_graph_dditable_ext_curr_t* _graph_ddi_table_ext = nullptr;

    ze_device_properties_t device_properties = {};

    uint32_t _group_ordinal;

    Logger log;
};
}  // namespace intel_npu
