// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "exceptions.hpp"

#include <sstream>

namespace ov {
namespace frontend {
namespace onnx_error {
namespace detail {
std::string get_error_msg_prefix(const ov::frontend::onnx::Node& node) {
    std::stringstream ss;
    ss << "While validating ONNX node '" << node << "'";
    return ss.str();
}
}  // namespace detail

void OnnxNodeValidationFailure::create(const char* file,
                                       int line,
                                       const char* check_string,
                                       const ov::frontend::onnx::Node& node,
                                       const std::string& explanation) {
    throw OnnxNodeValidationFailure(
        make_what(file, line, check_string, detail::get_error_msg_prefix(node), explanation));
}
}  // namespace onnx_error
}  // namespace frontend
}  // namespace ov
