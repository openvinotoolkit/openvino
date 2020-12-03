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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>

#include "lp_norm.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/norm.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/validation_util.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/exceptions.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector lp_norm(const Node& node)
                {
                    const Output<ngraph::Node> data{node.get_ng_inputs().at(0)};
                    const auto data_shape = data.get_partial_shape();
                    const auto data_rank = data_shape.rank();

                    CHECK_VALID_NODE(
                        node, data_shape.is_static(), "Data shape must be static for lp_norm op");
                    const auto data_rank_value = data_rank.get_length();
                    const std::int64_t p_norm{node.get_attribute_value<std::int64_t>("p", 2)};

                    const std::int64_t axis{node.get_attribute_value<std::int64_t>("axis", -1)};
                    const size_t normalize_axis =
                        ngraph::normalize_axis(node.get_description(), axis, data_rank);

                    CHECK_VALID_NODE(node,
                                     p_norm == 1 || p_norm == 2,
                                     "Invalid `p` attribute value: ",
                                     p_norm,
                                     "Only normalization of 1st or 2nd order is supported.");

                    const auto normalize_axis_const =
                        default_opset::Constant::create(element::Type_t::i64, {}, {normalize_axis});
                    std::shared_ptr<ngraph::Node> norm = ngraph::builder::opset1::lp_norm(
                        data, normalize_axis_const, static_cast<std::size_t>(p_norm));

                    const auto target_shape =
                        default_opset::Constant::create(element::Type_t::i64,
                                                        Shape{size_t(data_rank_value)},
                                                        data_shape.to_shape());

                    // Create a default axes order matching the data tensor rank and erase the
                    // element at the 'normalize_axis' position. The erased element indicates the
                    // axis
                    // along which the data should be broadcasted.
                    std::vector<size_t> axes_values(data_rank_value);
                    std::iota(axes_values.begin(), axes_values.end(), 0);
                    axes_values.erase(axes_values.begin() + normalize_axis);

                    const auto axes_mapping = default_opset::Constant::create(
                        element::Type_t::i64, Shape{axes_values.size()}, axes_values);

                    norm = std::make_shared<default_opset::Broadcast>(
                        norm, target_shape, axes_mapping);

                    return {std::make_shared<default_opset::Divide>(data, norm)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
