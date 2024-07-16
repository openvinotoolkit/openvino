// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>

#include "intel_npu/utils/logger/logger.hpp"
#include "npu.hpp"
#include "zero_init.hpp"

namespace intel_npu {
class ZeroEngineBackend final : public IEngineBackend {
public:
    ZeroEngineBackend(const Config& config);
    virtual ~ZeroEngineBackend();
    const std::shared_ptr<IDevice> getDevice() const override;
    const std::shared_ptr<IDevice> getDevice(const std::string&) const override;
    const std::string getName() const override {
        return "LEVEL0";
    }
    const std::vector<std::string> getDeviceNames() const override;
    uint32_t getDriverVersion() const override;
    uint32_t getDriverExtVersion() const override;

    bool isBatchingSupported() const override;
    bool isWorkloadTypeSupported() const override;

    void* getContext() const override;
    void updateInfo(const Config& config) {
        _logger.setLevel(config.get<LOG_LEVEL>());
        if (_devices.size() > 0) {
            for (auto& dev : _devices) {
                dev.second->updateInfo(config);
            }
        }
    }

private:
    std::shared_ptr<ZeroInitStructsHolder> _instance;

    std::map<std::string, std::shared_ptr<IDevice>> _devices{};
    Logger _logger;
};

}  // namespace intel_npu
