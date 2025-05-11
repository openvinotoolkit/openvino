// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iterator>
#include <sstream>
#include <string>

#include "openvino/core/node.hpp"

namespace ov {
namespace op {
namespace validate {
/**
 * @brief Provides `NodeValidationFailure` exception explanation string.
 *
 * @param shapes       Vector of shapes used for inference to be printed before explanation.
 * @param explanation  String with exception explanation.
 * @return             Explanation string.
 */
template <class TShape>
std::string shape_infer_explanation_str(const std::vector<TShape>& shapes, const std::string& explanation) {
    std::stringstream o;
    o << "Shape inference input shapes {";
    std::copy(shapes.cbegin(), std::prev(shapes.cend()), std::ostream_iterator<TShape>(o, ","));
    if (!shapes.empty()) {
        o << shapes.back();
    }
    o << "}\n" << explanation;
    return o.str();
}
}  // namespace validate
}  // namespace op
}  // namespace ov
