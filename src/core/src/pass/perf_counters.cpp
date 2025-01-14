// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "perf_counters.hpp"

namespace ov {
namespace pass {
openvino::itt::handle_t PerfCounters::operator[](ov::Node::type_info_t const& type_inf) {
    std::lock_guard<std::mutex> guard(m_mutex);
    auto it = m_counters.find(&type_inf);
    if (it != m_counters.end())
        return it->second;
    return m_counters[&type_inf] = openvino::itt::handle(type_inf.name);
}
}  // namespace pass
}  // namespace ov
