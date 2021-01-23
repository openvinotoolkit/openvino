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

#include <memory>

#include "default_opset.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/validation_util.hpp"
#include "op/softmax.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace
        {
            std::shared_ptr<ngraph::Node> onnx_softmax(const Output<ngraph::Node> data,
                                                       const int64_t axis)
            {
                const auto coerced_data = ngraph::builder::opset1::flatten(data, axis);
                const auto result = std::make_shared<default_opset::Softmax>(coerced_data, 1);
                const auto data_shape = std::make_shared<default_opset::ShapeOf>(data);
                const bool special_zero = false;
                return std::make_shared<default_opset::Reshape>(result, data_shape, special_zero);
            }
        }

        namespace op
        {
            namespace set_1
            {
                OutputVector softmax(const Node& node)
                {
                    const auto data = node.get_ng_inputs().at(0);
                    const auto data_rank = data.get_partial_shape().rank();
                    NGRAPH_CHECK(data_rank.is_static(),
                                 "ONNX Softmax data rank needs to be known (static)");

                    const auto axis = node.get_attribute_value<int64_t>("axis", 1);

                    std::shared_ptr<ngraph::Node> result;
                    switch (data_rank.get_length())
                    {
                    case 0:
                    {
                        result =
                            default_opset::Constant::create(data.get_element_type(), Shape{}, {1});
                        break;
                    }
                    case 1:
                    {
                        // checks if the axis belongs to the allowed values set (-1 and 0 for 1D)
                        ngraph::normalize_axis(
                            node.get_description(), axis, data.get_partial_shape().rank());
                        result = std::make_shared<default_opset::Softmax>(data, 0);
                        break;
                    }
                    default:
                    {
                        const auto normalized_axis = ngraph::normalize_axis(
                            node.get_description(), axis, data.get_partial_shape().rank());

                        result = onnx_softmax(data, normalized_axis);
                        break;
                    }
                    }

                    return {result};
                }
            }
        }
    }
}
