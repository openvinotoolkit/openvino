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

#include "mean.hpp"
#include "default_opset.hpp"
#include "utils/variadic.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector mean(const Node& node)
                {
                    auto sum = variadic::make_ng_variadic_op<default_opset::Add>(node).front();
                    auto count = default_opset::Constant::create(
                        sum->get_element_type(), Shape{}, {node.get_ng_inputs().size()});

                    return {std::make_shared<default_opset::Divide>(sum, count)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
