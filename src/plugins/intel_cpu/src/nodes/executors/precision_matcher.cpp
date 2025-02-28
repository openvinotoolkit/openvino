// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_matcher.hpp"

#include <cassert>

#include "nodes/executors/precision_translation.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

bool match(const InOutTypeMask& patterns, const InOutTypes& values) {
    assert(patterns.size() == values.size());

    return std::equal(values.begin(),
                      values.end(),
                      patterns.begin(),
                      [](const ov::element::Type value, const TypeMask pattern) {
                          return pattern & value;
                      });

    return true;
}

}  // namespace ov::intel_cpu
