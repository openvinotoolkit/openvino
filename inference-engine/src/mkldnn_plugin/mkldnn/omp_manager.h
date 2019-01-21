// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Abstraction over platform specific implementations
 * @file omp_manager.h
 */
#pragma once

namespace MKLDNNPlugin {
namespace cpu {

bool checkOpenMpEnvVars(bool includeOMPNumThreads = true);
// numbers of CPU sockets in the machine (on Linux), 1 on all other OSes
int getNumberOfCPUSockets();
// numbers of CPU physical cores on Linux (which is considered to be more performance friendly for servers)
// (on other OSes it simply relies on the original parallel API of choice, which usually use the logical cores )
int getNumberOfCPUCores();

}  // namespace cpu
}  // namespace MKLDNNPlugin