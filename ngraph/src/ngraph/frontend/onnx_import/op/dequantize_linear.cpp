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
#include "ngraph/opsets/opset0.hpp"
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
                    if (inputs.size() == 3 && !inputs.at(2)->is_null())
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

                constexpr const int64_t DEFAULT_AXIS = 1;

                // void validate_input(const char* input_name,
                //                     const NodeVector& inputs,
                //                     const size_t validated_input_index,
                //                     const int64_t axis,
                //                     const std::set<element::Type> allowed_types)
                // {
                //     const auto& validated_input = inputs[validated_input_index];
                //     const auto validated_input_rank =
                //         validated_input->get_output_partial_shape(0).rank();

                //     NGRAPH_CHECK(validated_input_rank.same_scheme({0}) ||
                //                      validated_input_rank.same_scheme({1}),
                //                  input_name,
                //                  " needs to be a scalar or a vector.");

                //     if (validated_input_rank.same_scheme({1}))
                //     {
                //         const auto x_input_shape = inputs[0]->get_output_partial_shape(0);
                //         const auto dim_at_axis = x_input_shape[axis];

                //         NGRAPH_CHECK(dim_at_axis.compatible(validated_input_rank.get_length()),
                //                      "The number of ",
                //                      input_name,
                //                      " elements has to match the dimension value pointed to by
                //                      the "
                //                      "axis attribute.");
                //     }

                //     const bool data_type_ok =
                //         allowed_types.count(validated_input->get_element_type());

                //     NGRAPH_CHECK(data_type_ok,
                //                  "Incorrect data type of the ",
                //                  input_name,
                //                  " input: ",
                //                  input->get_element_type());
                // }
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

                // namespace set_13
                // {
                    // NodeVector dequantize_linear(const Node& node)
                    // {
                    //     const NodeVector inputs{node.get_ng_inputs()};

                    //     NGRAPH_CHECK(
                    //         2 <= inputs.size() <= 3,
                    //         "The DequantizeLinear op expects 2 required and one optional input.
                    //         Got:
                    //         ",
                    //         inputs.size());

                    //     const std::shared_ptr<ngraph::Node> x = inputs[0];
                    //     const std::shared_ptr<ngraph::Node> scale = inputs[1];
                    //     const std::shared_ptr<ngraph::Node> zero_point = get_zero_point(inputs);

                    //     const auto x_rank = x->get_output_partial_shape(0).rank();

                    //     NGRAPH_CHECK(x_rank.is_static(),
                    //                  "Rank of the input data tensor has to be known/static.");

                    //     int64_t axis{node.get_attribute_value<int64_t>("axis", 1)};
                    //     axis = ngraph::normalize_axis(node.get_description(), axis, x_rank);

                    //     validate_input("Dequantization scale", scale, axis, {element::f32});
                    //     validate_input(
                    //         "Zero point", zero_point, axis, {element::i8, element::u8,
                    //         element::i32});

                    //     if (x->get_element_type() != zero_point->get_element_type())
                    //     {
                    //         zero_point = std::make_shared<default_opset::Convert>(
                    //             zero_point, x->get_element_type());
                    //     }

                    //     return {std::make_shared<ngraph::opset0::Dequantize>(
                    //         x, x_scale, zero_point, x_scale->get_element_type(), axes)};
                    // }
                // }
            }
        }
    }
}
