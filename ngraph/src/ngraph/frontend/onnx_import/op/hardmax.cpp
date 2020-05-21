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

#include "hardmax.hpp"
#include "exceptions.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/opsets/opset0.hpp"
#include "ngraph/validation_util.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector hardmax(const Node& node)
                {
                    const auto input = node.get_ng_inputs().at(0);
                    const auto& input_shape = input->get_output_partial_shape(0);

                    auto axis = node.get_attribute_value<std::int64_t>("axis", 1);
                    if (input_shape.rank().is_static())
                    {
                        axis = ngraph::normalize_axis(
                            node.get_description(), axis, input_shape.rank());
                    }

                    // reshape to 2D - "batch size" x "input feature dimensions" (NxD)
                    const auto coerced_tensor = ngraph::builder::opset1::flatten(input, axis);

                    const auto coerced_tensor_shape =
                        std::make_shared<default_opset::ShapeOf>(coerced_tensor);
                    std::shared_ptr<ngraph::Node> row_size =
                        std::make_shared<default_opset::Gather>(
                            coerced_tensor_shape,
                            default_opset::Constant::create(element::i64, {1}, {1}),
                            default_opset::Constant::create(element::i64, {}, {0}));
                    row_size = ngraph::onnx_import::reshape::interpret_as_scalar(row_size);

                    const auto indices_axis = 1;
                    const auto max_indices = std::make_shared<opset0::GetOutputElement>(
                        std::make_shared<default_opset::TopK>(
                            coerced_tensor,
                            default_opset::Constant::create(ngraph::element::i64, Shape{}, {1}),
                            indices_axis,
                            default_opset::TopK::Mode::MAX,
                            default_opset::TopK::SortType::NONE),
                        1);

                    const auto on_value =
                        default_opset::Constant::create(ngraph::element::i64, Shape{}, {1});
                    const auto off_value =
                        default_opset::Constant::create(ngraph::element::i64, Shape{}, {0});

                    const auto results = std::make_shared<default_opset::OneHot>(
                        max_indices, row_size, on_value, off_value, indices_axis);
                    const auto converted_results = std::make_shared<default_opset::Convert>(
                        results, input->get_element_type());

                    if (input_shape.is_static())
                    {
                        return {ngraph::builder::opset1::reshape(converted_results,
                                                                 input_shape.to_shape())};
                    }
                    else
                    {
                        const auto output_shape = std::make_shared<default_opset::ShapeOf>(input);
                        return {
                            std::make_shared<default_opset::Reshape>(input, output_shape, false)};
                    }
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
