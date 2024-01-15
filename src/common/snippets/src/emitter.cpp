// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/emitter.hpp"

namespace ov {
namespace snippets {

std::string regTypeToStr(const RegType& type) {
    switch (type) {
        case RegType::vec:
            return "vec";
        case RegType::gpr:
            return "gpr";
    }
    return "undefined";
}

}  // namespace snippets
}  // namespace ov
