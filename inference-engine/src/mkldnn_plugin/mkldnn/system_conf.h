// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Abstraction over platform specific implementations
 * @file system_conf.h
 */
#pragma once

#include <vector>
namespace MKLDNNPlugin {
namespace cpu {

bool checkOpenMpEnvVars(bool includeOMPNumThreads = true);
// available CPU NUMA nodes (on Linux, and Windows <only with TBB>, single node is assumed on all other OSes)
std::vector<int> getAvailableNUMANodes();
// numbers of CPU physical cores on Linux/Windows (which is considered to be more performance friendly for servers)
// (on other OSes it simply relies on the original parallel API of choice, which usually uses the logical cores )
int getNumberOfCPUCores();

}  // namespace cpu
}  // namespace MKLDNNPlugin