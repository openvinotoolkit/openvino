// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "singleton.hpp"
std::list<std::shared_ptr<SingleMem>> SingleMem::m_pointers[MAX_UTILS_SUPPORTED_PRIORITY];
std::mutex SingleMem::m_mutex;

void SingleMem::releaseSingltons() {
    std::lock_guard<std::mutex> lockGuard(m_mutex);
    for (int i = MAX_UTILS_SUPPORTED_PRIORITY - 1; i >= 0; i--) {
        while (!m_pointers[i].empty()) {
            m_pointers[i].pop_front();
        }
    }
}
