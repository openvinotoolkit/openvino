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
#include <vector>

#include "ngraph/shape.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/op/softsign.hpp"

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
