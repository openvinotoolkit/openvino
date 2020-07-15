//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <string>

#include "core/node.hpp"
#include "ngraph/assertion.hpp"
#include "ngraph/check.hpp"
#include "ngraph/except.hpp"

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

        } // namespace  error

    } // namespace  onnx_import

} // namespace  ngraph

#define CHECK_VALID_NODE(node_, cond_, ...)                                                        \
    NGRAPH_CHECK_HELPER(                                                                           \
        ::ngraph::onnx_import::error::OnnxNodeValidationFailure, (node_), (cond_), ##__VA_ARGS__)
