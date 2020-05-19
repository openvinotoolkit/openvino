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
#include "ngraph/opsets/opset0.hpp"
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
                    const std::shared_ptr<ngraph::Node> data{node.get_ng_inputs().at(0)};
                    Output<ngraph::Node> scale(node.get_ng_inputs().at(1));
                    Output<ngraph::Node> bias(node.get_ng_inputs().at(2));
                    const Shape& data_shape = data->get_shape();
                    const Shape& scale_shape = scale.get_shape();
                    const Shape& bias_shape = bias.get_shape();
                    const float epsilon{node.get_attribute_value<float>("epsilon", 1e-5f)};

                    CHECK_VALID_NODE(
                        node,
                        (scale_shape.size() == 1 && scale_shape[0] == data_shape.at(1)),
                        "Scale input must be one dimensional vector of number of "
                        "input data channels size.");

                    CHECK_VALID_NODE(node,
                                     (bias_shape.size() == 1 && bias_shape[0] == data_shape.at(1)),
                                     "Bias input must be one dimensional vector of number of "
                                     "input data channels size.");

                    // all dimensions except spatial/feature
                    const AxisSet reduction_axes{
                        common::get_monotonic_range<std::size_t>(data_shape.size(), 2)};

                    const std::shared_ptr<ngraph::Node> eps_node =
                        std::make_shared<default_opset::Constant>(
                            data->get_element_type(), data_shape, std::vector<float>{epsilon});

                    scale = ngraph::builder::opset1::make_broadcast(scale, data_shape, 1);
                    bias = ngraph::builder::opset1::make_broadcast(bias, data_shape, 1);

                    Output<ngraph::Node> mean = builder::opset1::mean(data, reduction_axes);
                    mean =
                        ngraph::builder::opset1::make_broadcast(mean, data_shape, reduction_axes);

                    Output<ngraph::Node> variance = builder::opset1::variance(data, reduction_axes);
                    variance = ngraph::builder::opset1::make_broadcast(
                        variance, data_shape, reduction_axes);

                    const auto sqrt = std::make_shared<default_opset::Sqrt>(
                        std::make_shared<default_opset::Add>(variance, eps_node));

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
