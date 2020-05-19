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

#include "atanh.hpp"
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
                NodeVector atanh(const Node& node)
                {
                    std::shared_ptr<ngraph::Node> data{node.get_ng_inputs().at(0)};

                    // Define inverse hyperbolic tangent in terms of natural logarithm:
                    //
                    // atanh(x) = 0.5 * ln((1 + x) / (1 - x))
                    //

                    const auto one =
                        default_opset::Constant::create(data->get_element_type(), {}, {1.f});

                    const auto half =
                        default_opset::Constant::create(data->get_element_type(), {}, {0.5f});

                    const auto one_plus_x = std::make_shared<default_opset::Add>(one, data);
                    const auto one_minus_x = std::make_shared<default_opset::Subtract>(one, data);
                    const auto log_args =
                        std::make_shared<default_opset::Divide>(one_plus_x, one_minus_x);
                    const auto log_node = std::make_shared<default_opset::Log>(log_args);

                    return {std::make_shared<default_opset::Multiply>(half, log_node)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
