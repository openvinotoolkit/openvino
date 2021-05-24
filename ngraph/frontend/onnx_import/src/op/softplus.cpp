// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "op/softplus.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector softplus(const Node& node)
                {
                    const auto data = node.get_ng_inputs().at(0);
                    return {std::make_shared<default_opset::SoftPlus>(data)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
