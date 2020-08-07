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
#include <vector>

#include "ngraph/partial_shape.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            std::string concat_strings(
                const std::vector<std::reference_wrapper<const std::string>>& strings);

            std::string build_input_provenance_tag(const std::string& input_name,
                                                   const PartialShape& shape);

            std::string build_op_provenance_tag(const Node& onnx_node);

        } // namespace detail
    }     // namespace onnx_import
} // namespace ngraph
