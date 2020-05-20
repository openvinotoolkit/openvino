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

#include <memory>

#include "acosh.hpp"
#include "default_opset.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector acosh(const Node& node)
                {
                    std::shared_ptr<ngraph::Node> data{node.get_ng_inputs().at(0)};

                    // Define inverse hyperbolic cosine in terms of natural logarithm:
                    //
                    // arccosh(x) = ln(x + sqrt(x^2 - 1))
                    //

                    const auto one =
                        default_opset::Constant::create(data->get_element_type(), {}, {1.f});

                    const auto x_square = std::make_shared<default_opset::Multiply>(data, data);
                    const auto sqrt_args = std::make_shared<default_opset::Subtract>(x_square, one);
                    const auto sqrt_node = std::make_shared<default_opset::Sqrt>(sqrt_args);
                    const auto log_args = std::make_shared<default_opset::Add>(data, sqrt_node);

                    return {std::make_shared<default_opset::Log>(log_args)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
