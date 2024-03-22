//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <map>
#include <memory>

#include "vpux.hpp"
#include "zero_init.h"

namespace vpux {
class ZeroEngineBackend final : public vpux::IEngineBackend {
public:
    ZeroEngineBackend(const Config& config);
    virtual ~ZeroEngineBackend();
    const std::shared_ptr<IDevice> getDevice() const override;
    const std::shared_ptr<IDevice> getDevice(const std::string&) const override;
    const std::string getName() const override {
        return "LEVEL0";
    }
    const std::vector<std::string> getDeviceNames() const override;

private:
    std::shared_ptr<ZeroInitStructsHolder> _instance;

    std::map<std::string, std::shared_ptr<IDevice>> _devices{};
};

}  // namespace vpux
