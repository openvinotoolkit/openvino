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

#include "cum_sum.hpp"
#include "onnx_import/default_opset.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector cum_sum(const Node& node)
                {
                    auto inputs = node.get_ng_inputs();
                    auto data = inputs.at(0);
                    bool exclusive = node.get_attribute_value<std::int64_t>("exclusive", 0);
                    bool reverse = node.get_attribute_value<std::int64_t>("reverse", 0);
                    Output<ngraph::Node> axis;

                    if (inputs.size() > 1)
                    {
                        axis = inputs.at(1); // optional input, 0-D tensor
                    }
                    else
                    {
                        axis = default_opset::Constant::create(
                            element::Type_t::i64, Shape{}, {0}); // default
                    }
                    return OutputVector{
                        std::make_shared<default_opset::CumSum>(data, axis, exclusive, reverse)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
