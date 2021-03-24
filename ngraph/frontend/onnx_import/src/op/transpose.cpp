// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "ngraph/builder/reshape.hpp"
#include "ngraph/node.hpp"
#include "op/transpose.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector transpose(const Node& node)
                {
                    Output<ngraph::Node> data = node.get_ng_inputs().at(0);

                    auto permute_axes =
                        node.get_attribute_value<std::vector<std::size_t>>("perm", {});

                    return {(permute_axes.empty())
                                ? ngraph::builder::opset1::transpose(data)
                                : ngraph::builder::opset1::reorder_axes(data, permute_axes)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
