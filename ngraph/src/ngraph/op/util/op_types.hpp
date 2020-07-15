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

#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"

namespace ngraph {
    namespace op {
        namespace util {
            NGRAPH_API
            bool is_unary_elementwise_arithmetic(const ngraph::Node* node);
            NGRAPH_API
            bool is_binary_elementwise_arithmetic(const ngraph::Node* node);
            NGRAPH_API
            bool is_binary_elementwise_comparison(const ngraph::Node* node);
            NGRAPH_API
            bool is_binary_elementwise_logical(const ngraph::Node* node);

            NGRAPH_API
            bool supports_auto_broadcast(const ngraph::Node* node);

            NGRAPH_API
            bool supports_decompose(const ngraph::Node* node);

            NGRAPH_API
            bool is_op(const ngraph::Node* node);
            NGRAPH_API
            bool is_parameter(const ngraph::Node* node);
        }
    }
}
