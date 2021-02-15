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

#include "core/null_node.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/max_pool.hpp"
#include "op/max_pool.hpp"
#include "utils/pooling_factory.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector max_pool(const Node& node)
                {
                    if (node.get_outputs_size() > 1)
                    {
                        NGRAPH_WARN
                            << "Indices output is not supported for MaxPooling and was ignored";
                    }
                    auto max_pool = pooling::PoolingFactory(node).make_max_pool();
                    max_pool.emplace_back(std::make_shared<NullNode>()); // Indices (optional)
                    return max_pool;
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
