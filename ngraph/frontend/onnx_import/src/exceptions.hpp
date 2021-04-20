// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "ngraph/check.hpp"
#include "ngraph/except.hpp"
#include "onnx_import/core/node.hpp"
#include "utils/tensor_external_data.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace error
        {
            namespace detail
            {
                std::string get_error_msg_prefix(const Node& node);
            }

            class OnnxNodeValidationFailure : public CheckFailure
            {
            public:
                OnnxNodeValidationFailure(const CheckLocInfo& check_loc_info,
                                          const Node& node,
                                          const std::string& explanation)
                    : CheckFailure(check_loc_info, detail::get_error_msg_prefix(node), explanation)
                {
                }
            };

            struct invalid_external_data : ngraph_error
            {
                invalid_external_data(const onnx_import::detail::TensorExternalData& external_data)
                    : ngraph_error{std::string{"invalid external data: "} +
                                   external_data.to_string()}
                {
                }
            };

        } // namespace  error

    } // namespace  onnx_import

} // namespace  ngraph

#define CHECK_VALID_NODE(node_, cond_, ...)                                                        \
    NGRAPH_CHECK_HELPER(                                                                           \
        ::ngraph::onnx_import::error::OnnxNodeValidationFailure, (node_), (cond_), ##__VA_ARGS__)
