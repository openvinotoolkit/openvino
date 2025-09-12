// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <iostream>

#include "openvino/core/except.hpp"

namespace ov::snippets::lowered {

enum class SpecificLoopIterType : uint8_t { FIRST_ITER, MAIN_BODY, LAST_ITER };

inline std::ostream& operator<<(std::ostream& out, const SpecificLoopIterType& type) {
    switch (type) {
    case SpecificLoopIterType::FIRST_ITER:
        out << "FIRST_ITER";
        break;
    case SpecificLoopIterType::MAIN_BODY:
        out << "MAIN_BODY";
        break;
    case SpecificLoopIterType::LAST_ITER:
        out << "LAST_ITER";
        break;
    default:
        OPENVINO_THROW("Unknown SpecificLoopIterType");
    }
    return out;
}

}  // namespace ov::snippets::lowered
