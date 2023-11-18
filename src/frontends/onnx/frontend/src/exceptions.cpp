// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "exceptions.hpp"

#include <sstream>

#include "openvino/core/deprecated.hpp"

namespace ngraph {
namespace onnx_import {
namespace error {
namespace detail {
OPENVINO_SUPPRESS_DEPRECATED_START
std::string get_error_msg_prefix(const Node& node) {
    std::stringstream ss;
    ss << "While validating ONNX node '" << node << "'";
    return ss.str();
}
}  // namespace detail

void OnnxNodeValidationFailure::create(const char* file,
                                       int line,
                                       const char* check_string,
                                       const Node& node,
                                       const std::string& explanation) {
    throw OnnxNodeValidationFailure(
        make_what({file, line, check_string}, detail::get_error_msg_prefix(node), explanation));
}
OPENVINO_SUPPRESS_DEPRECATED_END
}  // namespace error
}  // namespace onnx_import
}  // namespace ngraph
