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
#include <memory>

#include "ngraph/type/element_type.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/op/cast.hpp"
#include "onnx_import/utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector cast(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    int64_t target_type = node.get_attribute_value<int64_t>("to");
                    element::Type elem_type = common::get_ngraph_element_type(target_type);

                    return {std::make_shared<default_opset::Convert>(data, elem_type)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
