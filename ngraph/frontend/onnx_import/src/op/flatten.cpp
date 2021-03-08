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

#include <cinttypes>

#include "exceptions.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/validation_util.hpp"
#include "op/flatten.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector flatten(const Node& node)
                {
                    OutputVector inputs{node.get_ng_inputs()};
                    auto data = inputs.at(0);
                    auto axis = node.get_attribute_value<std::int64_t>("axis", 1);
                    const auto data_rank = data.get_partial_shape().rank();

                    if (data_rank.is_static())
                    {
                        const std::int64_t data_rank_value = data_rank.get_length();
                        // Accepted range is [-r, r] where r = rank(input).
                        axis = ngraph::normalize_axis(node.get_description(),
                                                      axis,
                                                      data_rank_value,
                                                      -data_rank_value,
                                                      data_rank_value);
                    }
                    return {ngraph::builder::opset1::flatten(data, axis)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace  onnx_import

} // namespace  ngraph
