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

#include <cstddef>
#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "instance_norm.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/reduce_ops.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/partial_shape.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector instance_norm(const Node& node)
                {
                    Output<ngraph::Node> data(node.get_ng_inputs().at(0));
                    Output<ngraph::Node> scale(node.get_ng_inputs().at(1));
                    Output<ngraph::Node> bias(node.get_ng_inputs().at(2));
                    const PartialShape& data_pshape = data.get_partial_shape();
                    const PartialShape& scale_pshape = scale.get_partial_shape();
                    const PartialShape& bias_pshape = bias.get_partial_shape();
                    const float epsilon{node.get_attribute_value<float>("epsilon", 1e-5f)};

                    element::Type result_et;
                    CHECK_VALID_NODE(
                        node,
                        element::Type::merge(
                            result_et, data.get_element_type(), scale.get_element_type()),
                        "Element types for data and scale input do not match (data element type: ",
                        data.get_element_type(),
                        ", scale element type: ",
                        scale.get_element_type(),
                        ").");

                    CHECK_VALID_NODE(
                        node,
                        element::Type::merge(
                            result_et, data.get_element_type(), bias.get_element_type()),
                        "Element types for data and bias input do not match (data element type: ",
                        data.get_element_type(),
                        ", bias element type: ",
                        bias.get_element_type(),
                        ").");

                    CHECK_VALID_NODE(node,
                                     (scale_pshape.rank().is_static() &&
                                      scale_pshape.rank().get_length() == 1 &&
                                      data_pshape.rank().is_static() &&
                                      data_pshape[1].same_scheme(scale_pshape[0])),
                                     "Scale input must be one dimensional vector of number of "
                                     "input data channels size.");

                    CHECK_VALID_NODE(node,
                                     (bias_pshape.rank().is_static() &&
                                      bias_pshape.rank().get_length() == 1 &&
                                      data_pshape.rank().is_static() &&
                                      data_pshape[1].same_scheme(bias_pshape[0])),
                                     "Bias input must be one dimensional vector of number of "
                                     "input data channels size.");

                    // all dimensions except spatial/feature
                    const auto reduction_axes =
                        common::get_monotonic_range_along_node_rank(data, 2);

                    const std::shared_ptr<ngraph::Node> eps_node =
                        std::make_shared<default_opset::Constant>(
                            data.get_element_type(), Shape{}, epsilon);

                    Output<ngraph::Node> mean = builder::opset1::mean(data, reduction_axes, true);
                    Output<ngraph::Node> variance =
                        builder::opset1::variance(data, reduction_axes, true);

                    const auto sqrt = std::make_shared<default_opset::Sqrt>(
                        std::make_shared<default_opset::Add>(variance, eps_node));

                    std::shared_ptr<ngraph::Node> data_shape_node;
                    if (data_pshape.is_static())
                    {
                        data_shape_node = std::make_shared<default_opset::Constant>(
                            element::i64,
                            Shape{static_cast<size_t>(data_pshape.rank().get_length())},
                            data_pshape.to_shape());
                    }
                    else
                    {
                        data_shape_node = std::make_shared<default_opset::ShapeOf>(data);
                    }

                    // Broadcast preserving channel dimension
                    scale = std::make_shared<default_opset::Broadcast>(
                        scale,
                        data_shape_node,
                        std::make_shared<default_opset::Constant>(element::i64, Shape{1}, 1));
                    bias = std::make_shared<default_opset::Broadcast>(
                        bias,
                        data_shape_node,
                        std::make_shared<default_opset::Constant>(element::i64, Shape{1}, 1));

                    // scale * (data - mean) / sqrt + bias
                    std::shared_ptr<ngraph::Node> result{
                        std::make_shared<default_opset::Subtract>(data, mean)};
                    result = std::make_shared<default_opset::Multiply>(scale, result);
                    result = std::make_shared<default_opset::Divide>(result, sqrt);
                    result = std::make_shared<default_opset::Add>(result, bias);

                    return {result};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
