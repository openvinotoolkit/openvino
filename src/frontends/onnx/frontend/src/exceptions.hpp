// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "ngraph/check.hpp"
#include "ngraph/except.hpp"
#include "onnx_import/core/node.hpp"
#include "openvino/core/except.hpp"
#include "utils/tensor_external_data.hpp"

namespace ngraph {
namespace onnx_import {
namespace error {
namespace detail {
std::string get_error_msg_prefix(const Node& node);
}

class OnnxNodeValidationFailure : public ov::AssertFailure {
public:
    [[noreturn]] static void create(const CheckLocInfo& check_loc_info,
                                    const Node& node,
                                    const std::string& explanation);

protected:
    explicit OnnxNodeValidationFailure(const std::string& what_arg) : ov::AssertFailure(what_arg) {}
};

OPENVINO_SUPPRESS_DEPRECATED_START
struct invalid_external_data : ngraph_error {
    invalid_external_data(const onnx_import::detail::TensorExternalData& external_data)
        : ngraph_error{std::string{"invalid external data: "} + external_data.to_string()} {}
    invalid_external_data(const std::string& what_arg) : ngraph_error{what_arg} {}
};
OPENVINO_SUPPRESS_DEPRECATED_END

}  // namespace  error

}  // namespace  onnx_import

}  // namespace  ngraph

#define CHECK_VALID_NODE(node_, cond_, ...) \
    OPENVINO_ASSERT_HELPER(::ngraph::onnx_import::error::OnnxNodeValidationFailure, (node_), (cond_), ##__VA_ARGS__)
