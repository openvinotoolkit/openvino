// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "core/null_node.hpp"
#include "ngraph/log.hpp"
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
                        NGRAPH_WARN << "MaxPool: Indices output is not supported and was ignored";
                    }
                    auto max_pool = pooling::PoolingFactory(node).make_max_pool();
                    max_pool.emplace_back(std::make_shared<NullNode>()); // Indices (optional)
                    return max_pool;
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph