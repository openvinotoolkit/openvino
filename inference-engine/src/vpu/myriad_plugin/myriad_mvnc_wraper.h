// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <vector>
#include <memory>
#include <string>
#include <mvnc.h>

namespace vpu {
namespace MyriadPlugin {

//------------------------------------------------------------------------------
// class IMvnc
// This is a class interface for accessing devices.
//------------------------------------------------------------------------------

class IMvnc {
public:
    // Operations
    virtual std::vector<ncDeviceDescr_t> AvailableDevicesDesc() const = 0;
    virtual std::vector<std::string> AvailableDevicesNames() const = 0;

    // Destructor
    virtual ~IMvnc() = default;
};

//------------------------------------------------------------------------------
// class Mvnc
// This is a wrapper of mvnc library.
//------------------------------------------------------------------------------

class Mvnc : public IMvnc {
public:
    // Operations
    std::vector<ncDeviceDescr_t> AvailableDevicesDesc() const override;
    std::vector<std::string> AvailableDevicesNames() const override;
};

}  // namespace MyriadPlugin
}  // namespace vpu
