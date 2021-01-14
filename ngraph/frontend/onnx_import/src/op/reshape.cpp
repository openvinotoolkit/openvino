//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <cstddef>
#include <memory>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/shape.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/exceptions.hpp"
#include "onnx_import/op/reshape.hpp"
#include "onnx_import/utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector reshape(const Node& node)
                {
                    OutputVector ng_inputs{node.get_ng_inputs()};
                    const auto data = ng_inputs.at(0);

                    Output<ngraph::Node> pattern;

                    // Since opset 5 the target shape is provided as input
                    if (ng_inputs.size() == 2)
                    {
                        pattern = ng_inputs.at(1);
                    }
                    else
                    {
                        const auto output_shape =
                            node.get_attribute_value<std::vector<int64_t>>("shape", {});

                        pattern = default_opset::Constant::create(
                            element::i64, Shape{output_shape.size()}, output_shape);
                    }

                    return {std::make_shared<default_opset::Reshape>(data, pattern, true)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
