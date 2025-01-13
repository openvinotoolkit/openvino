// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/emitter.hpp"

namespace ov {
namespace snippets {

bool operator==(const Reg& lhs, const Reg& rhs) {
    return lhs.type == rhs.type && lhs.idx == rhs.idx;
}
bool operator!=(const Reg& lhs, const Reg& rhs) {
    return !(lhs == rhs);
}

std::string regTypeToStr(const RegType& type) {
    switch (type) {
        case RegType::vec:
            return "vec";
        case RegType::gpr:
            return "gpr";
        default:
            OPENVINO_THROW("Unexpected RegType");
    }
}

}  // namespace snippets
}  // namespace ov
