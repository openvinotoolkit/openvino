// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "default_opset.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/multiply.hpp"
#include "op/expand.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector expand(const Node& node)
                {
                    const Output<ngraph::Node> data{node.get_ng_inputs().at(0)};
                    const Output<ngraph::Node> shape{node.get_ng_inputs().at(1)};

                    return {std::make_shared<default_opset::Broadcast>(
                        data, shape, ngraph::op::BroadcastType::BIDIRECTIONAL)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
