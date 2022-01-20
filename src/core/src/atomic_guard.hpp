// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <atomic>

namespace ov {

// The class AtomicGuard is an atomic wrapper that provides a convenient RAII-style mechanism for emulate mutex
class AtomicGuard {
public:
    AtomicGuard(std::atomic_bool& b) : m_atomic(b) {
        bool exp = false;
        while (m_atomic.load(std::memory_order_relaxed) || !m_atomic.compare_exchange_strong(exp, true)) {
            exp = false;
        }
    }
    ~AtomicGuard() {
        m_atomic = false;
    }

private:
    std::atomic_bool& m_atomic;
};

}  // namespace ov
