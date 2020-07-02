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

#include <cstdint>
#include <memory>

#include "default_opset.hpp"
#include "dequantize_linear.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/validation_util.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace
            {
                std::shared_ptr<ngraph::Node> get_zero_point(const NodeVector& inputs)
                {
                    if (inputs.size() == 3 && !inputs[2]->is_null())
                    {
                        auto zero_point = inputs[2];

                        if (zero_point->get_element_type() != element::f32)
                        {
                            zero_point =
                                std::make_shared<default_opset::Convert>(zero_point, element::f32);
                        }

                        return zero_point;
                    }
                    else
                    {
                        return default_opset::Constant::create(element::f32, Shape{}, {0});
                    }
                }
            }
            namespace set_1
            {
                namespace
                {
                    void validate_scalar_input(const char* input_name,
                                               const std::shared_ptr<ngraph::Node> input,
                                               const std::set<element::Type> allowed_types = {})
                    {
                        const auto validated_input_rank = input->get_output_partial_shape(0).rank();

                        NGRAPH_CHECK(validated_input_rank.same_scheme({0}),
                                     input_name,
                                     " needs to be a scalar.");

                        if (!allowed_types.empty())
                        {
                            const bool data_type_ok =
                                allowed_types.count(input->get_element_type());
                            NGRAPH_CHECK(data_type_ok,
                                         "Incorrect data type of the ",
                                         input_name,
                                         " input: ",
                                         input->get_element_type());
                        }
                    }
                }

                NodeVector dequantize_linear(const Node& node)
                {
                    const NodeVector inputs{node.get_ng_inputs()};

                    NGRAPH_CHECK(
                        2 <= inputs.size() && inputs.size() <= 3,
                        "The DequantizeLinear op expects 2 required and one optional input. Got: ",
                        inputs.size());

                    const auto x = inputs[0];
                    const auto scale = inputs[1];
                    const auto zero_point = get_zero_point(inputs);

                    validate_scalar_input("Dequantization scale", scale, {element::f32});
                    validate_scalar_input("Zero point", zero_point);

                    const auto converted_x =
                        std::make_shared<default_opset::Convert>(x, element::f32);

                    return {std::make_shared<default_opset::Multiply>(
                        std::make_shared<default_opset::Subtract>(converted_x, zero_point), scale)};
                }
            }

            namespace set_13
            {
                namespace
                {
                    void validate_scale(const std::shared_ptr<ngraph::Node> scale,
                                        const std::shared_ptr<ngraph::Node> x,
                                        const int64_t axis)
                    {
                        const auto& scale_shape = scale->get_output_partial_shape(0);
                        NGRAPH_CHECK(scale_shape.rank().get_length() == 0 ||
                                         scale_shape.rank().get_length() == 1,
                                     "Dequantization scale needs to be a scalar or a vector.");

                        if (scale_shape.rank().get_length() == 1)
                        {
                            const auto& scale_dim = scale_shape[0];
                            const auto& x_shape = x->get_output_partial_shape(0);
                            const auto& x_dim_at_axis = x_shape[axis];

                            NGRAPH_CHECK(scale_dim.same_scheme(x_dim_at_axis),
                                         "The number of dequantization scale elements '",
                                         scale_dim,
                                         "' must match the input shape dimension '",
                                         x_dim_at_axis,
                                         " pointed to by the axis attribute: ",
                                         axis);
                        }
                    }

                    void validate_zero_point(const std::shared_ptr<ngraph::Node> zero_point,
                                             const std::shared_ptr<ngraph::Node> x,
                                             const int64_t axis)
                    {
                        const auto& zero_point_shape = zero_point->get_output_partial_shape(0);
                        NGRAPH_CHECK(zero_point_shape.rank().get_length() == 0 ||
                                         zero_point_shape.rank().get_length() == 1,
                                     "Zero point needs to be a scalar or a vector.");

                        if (zero_point_shape.rank().get_length() == 1)
                        {
                            const auto& zero_point_dim = zero_point_shape[0];
                            const auto& x_shape = x->get_output_partial_shape(0);
                            const auto& x_dim_at_axis = x_shape[axis];

                            NGRAPH_CHECK(zero_point_dim.same_scheme(x_dim_at_axis),
                                         "The number of zero point elements '",
                                         zero_point_dim,
                                         "' must match the input shape dimension '",
                                         x_dim_at_axis,
                                         " pointed to by the axis attribute: ",
                                         axis);
                        }
                    }

                    std::shared_ptr<ngraph::Node>
                        reshape_input(const std::shared_ptr<ngraph::Node> input,
                                      const int64_t axis,
                                      const PartialShape& x_shape)
                    {
                        std::vector<int64_t> target_dims;

                        for (size_t i = 0; i < axis; ++i)
                        {
                            target_dims.push_back(1);
                        }

                        // copy dimension at axis from input X
                        target_dims.push_back(x_shape[axis].get_length());

                        for (size_t i = axis + 1; i < x_shape.rank().get_length(); ++i)
                        {
                            target_dims.push_back(1);
                        }

                        const auto target_shape = default_opset::Constant::create(
                            element::i64, Shape{target_dims.size()}, target_dims);

                        return std::make_shared<default_opset::Reshape>(input, target_shape, true);
                    }
                }

                NodeVector dequantize_linear(const Node& node)
                {
                    const NodeVector inputs{node.get_ng_inputs()};

                    NGRAPH_CHECK(2 <= inputs.size() && inputs.size() <= 3,
                                 "The DequantizeLinear op expects 2 required and one optional "
                                 "input. Got: ",
                                 inputs.size());

                    const auto x = inputs[0];
                    auto scale = inputs[1];
                    auto zero_point = get_zero_point(inputs);

                    const auto x_shape = x->get_output_partial_shape(0);

                    NGRAPH_CHECK(x_shape.rank().is_static(),
                                 "Rank of the input data tensor has to be known (static).");

                    int64_t axis{node.get_attribute_value<int64_t>("axis", 1)};
                    axis = ngraph::normalize_axis(node.get_description(), axis, x_shape.rank());

                    validate_scale(scale, x, axis);
                    validate_zero_point(zero_point, x, axis);

                    // these reshapes make sure that dequantization happens over the specified axis
                    scale = reshape_input(scale, axis, x_shape);
                    zero_point = reshape_input(zero_point, axis, x_shape);

                    const auto converted_x =
                        std::make_shared<default_opset::Convert>(x, element::f32);

                    return {std::make_shared<default_opset::Multiply>(
                        std::make_shared<default_opset::Subtract>(converted_x, zero_point), scale)};
                }
            }
        }
    }
}
