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

#include "exceptions.hpp"
#include "onnx_import/core/node.hpp"
#include "utils/arg_min_max_factory.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector argmax(const Node& node)
                {
                    const utils::ArgMinMaxFactory arg_factory(node);
                    return {arg_factory.make_arg_max()};
                }

            } // namespace set_1

            namespace set_12
            {
                OutputVector argmax(const Node& node)
                {
                    const auto select_last_index =
                        node.get_attribute_value<std::int64_t>("select_last_index", 0);
                    CHECK_VALID_NODE(node,
                                     select_last_index == 0,
                                     "Mode 'select_last_index=1' is not supported by current "
                                     "implementation of ArgMax");

                    const utils::ArgMinMaxFactory arg_factory(node);
                    return {arg_factory.make_arg_max()};
                }

            } // namespace set_12

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
