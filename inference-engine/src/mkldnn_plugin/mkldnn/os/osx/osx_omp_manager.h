// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

/**
* @brief WINAPI based code
* @file win_omp_manager.h
*/

#pragma once

#include <thread>
#include <vector>

namespace MKLDNNPlugin {
namespace cpu {

class OpenMpManager {
public:
    static int getOpenMpThreadNumber() {
        return getCoreNumber();
    }

    static int getCoreNumber() {
        return 4;
    }
};

}  // namespace cpu
}  // namespace MKLDNNPlugin

