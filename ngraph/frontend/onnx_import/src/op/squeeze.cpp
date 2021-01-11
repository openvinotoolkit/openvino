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

#include "ngraph/op/squeeze.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/validation_util.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/exceptions.hpp"
#include "onnx_import/op/squeeze.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector squeeze(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    std::vector<std::int64_t> axes =
                        node.get_attribute_value<std::vector<std::int64_t>>("axes", {});
                    const auto data_rank = data.get_partial_shape().rank();

                    std::vector<std::size_t> normalized_axes =
                        ngraph::normalize_axes(node.get_description(), axes, data_rank);
                    auto axes_node = std::make_shared<default_opset::Constant>(
                        element::u64, Shape{normalized_axes.size()}, normalized_axes);

                    return {std::make_shared<default_opset::Squeeze>(data, axes_node)};
                }

            } // namespace set_1

            namespace set_13
            {
                OutputVector squeeze(const Node& node)
                {
                    auto inputs = node.get_ng_inputs();
                    if (inputs.size() < 2)
                    {
                        std::vector<int64_t> axes{};
                        auto axes_node = std::make_shared<default_opset::Constant>(
                            element::Type_t::u64, Shape{}, axes);

                        return {std::make_shared<default_opset::Squeeze>(inputs.at(0), axes_node)};
                    }
                    else
                    {
                        return {
                            std::make_shared<default_opset::Squeeze>(inputs.at(0), inputs.at(1))};
                    }
                }

            } // namespace set_13
        }     // namespace op
    }         // namespace onnx_import
} // namespace ngraph
