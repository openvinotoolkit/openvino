// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "core/node.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/core/except.hpp"
#include "utils/tensor_external_data.hpp"

namespace ov {
namespace frontend {
namespace onnx_error {

namespace detail {
OPENVINO_SUPPRESS_DEPRECATED_START
std::string get_error_msg_prefix(const ov::frontend::onnx::Node& node);
OPENVINO_SUPPRESS_DEPRECATED_END
}  // namespace detail

class OnnxNodeValidationFailure : public ov::AssertFailure {
public:
    OPENVINO_SUPPRESS_DEPRECATED_START [[noreturn]] static void create(const char* file,
                                                                       int line,
                                                                       const char* check_string,
                                                                       const ov::frontend::onnx::Node& node,
                                                                       const std::string& explanation);
    OPENVINO_SUPPRESS_DEPRECATED_END

protected:
    explicit OnnxNodeValidationFailure(const std::string& what_arg) : ov::AssertFailure(what_arg) {}
};

OPENVINO_SUPPRESS_DEPRECATED_START
struct invalid_external_data : ov::Exception {
    invalid_external_data(const ov::frontend::onnx::detail::TensorExternalData& external_data)
        : ov::Exception{std::string{"invalid external data: "} + external_data.to_string()} {}
    invalid_external_data(const std::string& what_arg) : ov::Exception{what_arg} {}
};
OPENVINO_SUPPRESS_DEPRECATED_END

}  // namespace onnx_error
}  // namespace frontend
}  // namespace ov

namespace ov {
namespace frontend {
namespace onnx {
namespace error {
using namespace ov::frontend::onnx_error;
}  // namespace error
}  // namespace onnx
}  // namespace frontend
}  // namespace ov

#define CHECK_VALID_NODE(node_, cond_, ...) \
    OPENVINO_ASSERT_HELPER(ov::frontend::onnx_error::OnnxNodeValidationFailure, (node_), (cond_), ##__VA_ARGS__)
