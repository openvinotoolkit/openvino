// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>

namespace MKLDNNPlugin {

struct Config {
    bool useThreadBinding = true;
    bool collectPerfCounters = false;
    bool exclusiveAsyncRequests = false;
    bool enableDynamicBatch = false;
    int batchLimit = 0;

    void readProperties(const std::map<std::string, std::string> &config);
};

}  // namespace MKLDNNPlugin

