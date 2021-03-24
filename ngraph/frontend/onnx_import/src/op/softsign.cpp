// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "default_opset.hpp"
#include "ngraph/shape.hpp"
#include "op/softsign.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector softsign(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);

                    std::shared_ptr<ngraph::Node> one_node =
                        default_opset::Constant::create(data.get_element_type(), Shape{}, {1});
                    auto abs_data = std::make_shared<default_opset::Abs>(data);
                    auto data_plus_one_node =
                        std::make_shared<default_opset::Add>(abs_data, one_node);

                    return {std::make_shared<default_opset::Divide>(data, data_plus_one_node)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
