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
#include "onehot.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector onehot(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    auto indices =
                        std::make_shared<default_opset::Convert>(inputs.at(0), element::i64);
                    auto depth = reshape::interpret_as_scalar(inputs.at(1));

                    // Rank 1 tensor containing exactly two elements: [off_value, on_value]
                    auto values = inputs.at(2);
                    auto split_axis = default_opset::Constant::create(element::i64, {}, {0});
                    auto off_on_values =
                        std::make_shared<default_opset::Split>(values, split_axis, 2);
                    auto off_value =
                        reshape::interpret_as_scalar(get_output_element(off_on_values, size_t{0}));
                    auto on_value =
                        reshape::interpret_as_scalar(get_output_element(off_on_values, size_t{1}));

                    auto axis = node.get_attribute_value<std::int64_t>("axis", -1);

                    return {std::make_shared<default_opset::OneHot>(
                        indices, depth, on_value, off_value, axis)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace  onnx_import

} // namespace  ngraph
