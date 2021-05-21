// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "default_opset.hpp"
#include "op/prelu.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector prelu(const Node& node)
                {
                    OutputVector ng_inputs{node.get_ng_inputs()};
                    const auto& data = ng_inputs.at(0);
                    const auto& slope = ng_inputs.at(1);
                    return {std::make_shared<default_opset::PRelu>(data, slope)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
