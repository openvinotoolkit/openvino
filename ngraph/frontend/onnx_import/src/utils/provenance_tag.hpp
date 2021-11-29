// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
