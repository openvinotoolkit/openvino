// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace ov {
namespace util {
namespace monitor {
class PerformanceCounter {
public:
    PerformanceCounter(std::string deviceName) {}
    virtual std::map<std::string, double> get_load() = 0;
    std::string name() {
        return deviceName;
    }

private:
    std::string deviceName;
};
}  // namespace monitor
}  // namespace util
}  // namespace ov