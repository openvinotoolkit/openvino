// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

namespace ov {
namespace intel_cpu {

/**
 * Workaround for c++11 defect, where hashing support
 * is missed for enum classes
 */
struct EnumClassHash {
    template <typename T>
    int operator()(T t) const {
        return static_cast<int>(t);
    }
};

}  // namespace intel_cpu
}  // namespace ov
