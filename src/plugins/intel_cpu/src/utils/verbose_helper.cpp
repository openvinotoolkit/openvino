// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef CPU_DEBUG_CAPS
#    include "verbose_helper.h"

#    include <node.h>

#    include <cstdio>
#    include <cstdlib>
#    include <cstring>
#    include <iostream>
#    include <ostream>
#    include <sstream>

#    include "cpu_types.h"
#    include "utils/general_utils.h"

namespace ov::intel_cpu {

bool Verbose::shouldBePrinted() const {
    if (m_lvl < 1) {
        return false;
    }

    if (m_lvl < 2 && any_of(m_node->getType(), Type::Input, Type::Output)) {
        return false;
    }

    const bool low_level = m_lvl < 3;
    const bool is_constant = m_node->isConstant();
    const bool skip_node = low_level && is_constant;
    return !skip_node;
}

void Verbose::flush() const {
    std::cout << m_stream.rdbuf() << '\n';
}

}  // namespace ov::intel_cpu

#endif  // CPU_DEBUG_CAPS
