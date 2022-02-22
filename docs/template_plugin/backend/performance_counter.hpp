// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>

#include "ngraph/node.hpp"

namespace ngraph {
namespace runtime {
class PerformanceCounter {
public:
    PerformanceCounter(const std::shared_ptr<const Node>& n, size_t us, size_t calls)
        : m_node(n),
          m_total_microseconds(us),
          m_call_count(calls) {}
    std::shared_ptr<const Node> get_node() const {
        return m_node;
    }
    size_t total_microseconds() const {
        return m_total_microseconds;
    }
    size_t microseconds() const {
        return m_call_count == 0 ? 0 : m_total_microseconds / m_call_count;
    }
    size_t call_count() const {
        return m_call_count;
    }
    std::shared_ptr<const Node> m_node;
    size_t m_total_microseconds;
    size_t m_call_count;
};
}  // namespace runtime
}  // namespace ngraph
