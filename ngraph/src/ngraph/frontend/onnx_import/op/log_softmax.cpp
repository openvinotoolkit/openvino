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

#include <memory>

#include "default_opset.hpp"
#include "log_softmax.hpp"
#include "ngraph/validation_util.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector log_softmax(const Node& node)
                {
                    OutputVector inputs{node.get_ng_inputs()};
                    const auto data = inputs.at(0);
                    const auto data_rank = data.get_partial_shape().rank();

                    const auto axis = node.get_attribute_value<int64_t>("axis", 1);
                    const auto normalized_axis =
                        ngraph::normalize_axis(node.get_description(), axis, data_rank);

                    const auto softmax =
                        std::make_shared<default_opset::Softmax>(data, normalized_axis);
                    return {std::make_shared<default_opset::Log>(softmax)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
