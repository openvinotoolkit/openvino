// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mvnc.h>
#include <watchdog.h>

#include <functional>
#include <vector>
#include <memory>
#include <string>

namespace vpu {
namespace MyriadPlugin {

using WatchdogUniquePtr = std::unique_ptr<WatchdogHndl_t, std::function<void(WatchdogHndl_t*)>>;

//------------------------------------------------------------------------------
// class IMvnc
// This is a class interface for accessing devices.
//------------------------------------------------------------------------------

class IMvnc {
public:
    // Operations
    virtual std::vector<ncDeviceDescr_t> AvailableDevicesDesc() const = 0;
    virtual std::vector<std::string> AvailableDevicesNames() const = 0;

    virtual WatchdogHndl_t* watchdogHndl() = 0;

    // Destructor
    virtual ~IMvnc() = default;
};

//------------------------------------------------------------------------------
// class Mvnc
// This is a wrapper of mvnc library.
//------------------------------------------------------------------------------

class Mvnc : public IMvnc {
public:
    Mvnc();
    ~Mvnc() override = default;

    // Operations
    std::vector<ncDeviceDescr_t> AvailableDevicesDesc() const override;
    std::vector<std::string> AvailableDevicesNames() const override;

    WatchdogHndl_t* watchdogHndl() override {
        return m_watcdogPtr.get();
    }

private:
    WatchdogUniquePtr m_watcdogPtr;
};

}  // namespace MyriadPlugin
}  // namespace vpu
