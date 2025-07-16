// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_matcher.hpp"

#include <algorithm>

#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

bool match(const InOutTypeMask& patterns, const InOutTypes& values) {
    OPENVINO_DEBUG_ASSERT(patterns.size() == values.size(),
                          "Size of patterns must match size of values: patterns.size()=",
                          patterns.size(),
                          ", values.size()=",
                          values.size());

    return std::equal(values.begin(),
                      values.end(),
                      patterns.begin(),
                      [](const ov::element::Type value, const TypeMask pattern) {
                          return pattern & value;
                      });

    return true;
}

}  // namespace ov::intel_cpu
