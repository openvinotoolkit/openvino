// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Disabled in CMakeList
// Update to higher opset required

#include <memory>

#include "default_opset.hpp"
#include "op/scatter_nd.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector scatter_nd(const Node& node)
                {
                    OutputVector ng_inputs{node.get_ng_inputs()};
                    auto data = ng_inputs.at(0);
                    auto indices = ng_inputs.at(1);
                    auto updates = ng_inputs.at(2);

                    return {
                        std::make_shared<default_opset::ScatterNDUpdate>(data, indices, updates)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
