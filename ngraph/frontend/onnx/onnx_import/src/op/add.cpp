// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/add.hpp"
#include "default_opset.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/shape.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector add(const Node& node)
                {
                    return common::handle_opset6_binary_op<default_opset::Add>(node);
                }

            } // namespace set_1

            namespace set_7
            {
                OutputVector add(const Node& node)
                {
                    return {std::make_shared<default_opset::Add>(node.get_ng_inputs().at(0),
                                                                 node.get_ng_inputs().at(1))};
                }

            } // namespace set_7

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
