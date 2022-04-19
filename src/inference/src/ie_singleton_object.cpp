// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_singleton_object.hpp"

namespace InferenceEngine {
std::list<std::shared_ptr<SingletonMem>> SingletonMem::m_pointers[MAX_SUPPORTED_SINGLETON_PRIORITY];
std::mutex SingletonMem::m_mutex;

void SingletonMem::releaseSingltons() {
    std::lock_guard<std::mutex> lockGuard(m_mutex);
    for (int i = MAX_SUPPORTED_SINGLETON_PRIORITY - 1; i >= 0; i--) {
        while (!m_pointers[i].empty()) {
            m_pointers[i].pop_front();
        }
    }
}
}  // namespace InferenceEngine