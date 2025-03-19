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
bool operator<(const Reg& lhs, const Reg& rhs) {
    return lhs.type < rhs.type ||
           (lhs.type == rhs.type && lhs.idx < rhs.idx);
}
bool operator>(const Reg& lhs, const Reg& rhs) {
    return lhs.type > rhs.type ||
           (lhs.type == rhs.type && lhs.idx > rhs.idx);
}

std::ostream& operator<<(std::ostream& s, const Reg& r) {
    auto regTypeToStr = [](const RegType& type) {
        switch (type) {
            case RegType::vec:
                return "vec";
            case RegType::gpr:
                 return "gpr";
            case RegType::mask:
                return "mask";
            case RegType::undefined:
                 return "undefined";
            default:
                OPENVINO_THROW("Unexpected RegType");
        }
    };
    s << regTypeToStr(r.type) << "[" <<
        (r.idx == Reg::UNDEFINED_IDX ? "undefined" : std::to_string(r.idx))
      << "]";
    return s;
}

}  // namespace snippets
}  // namespace ov
