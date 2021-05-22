// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
