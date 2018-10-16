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
#include <windows.h>

namespace MKLDNNPlugin {
namespace cpu {

class OpenMpManager {
public:
    static int getOpenMpThreadNumber() {
        return getCoreNumber();
    }

    static int getCoreNumber() {
        int num_cores = std::thread::hardware_concurrency();
        unsigned long size = 0;

        if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &size)) {
            if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
                std::vector<char> buf(size);
                SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* info
                        = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(&buf.front());
                SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* ptr = info;
                if (GetLogicalProcessorInformationEx(RelationProcessorCore, info, &size)) {
                    if (GetLastError() == ERROR_SUCCESS) {
                        int num = 0;
                        unsigned long offset = 0;
                        while (offset < size) {
                            num++;
                            offset += ptr->Size;
                            ptr = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
                                    reinterpret_cast<byte*>(ptr) + ptr->Size);
                        }
                        num_cores = num;
                    }
                }
            }
        }
        return num_cores;
    }
};

}  // namespace cpu
}  // namespace MKLDNNPlugin

