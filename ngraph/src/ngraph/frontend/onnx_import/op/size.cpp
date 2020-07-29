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
#include <vector>

#include "default_opset.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "size.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector size(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    std::int64_t tensor_elements_count{
                        static_cast<std::int64_t>(shape_size(data.get_shape()))};

                    return {std::make_shared<default_opset::Constant>(
                        ngraph::element::i64,
                        Shape{},
                        std::vector<std::int64_t>{tensor_elements_count})};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
