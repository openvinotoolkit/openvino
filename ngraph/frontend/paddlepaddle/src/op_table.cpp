// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/conv2d.hpp"
#include "op/elementwise_ops.hpp"
#include "op/relu.hpp"
#include "op/scale.hpp"
#include "op/split.hpp"

#include "op_table.hpp"

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            std::map<std::string, CreatorFunction> get_supported_ops()
            {
                return {{"conv2d", op::conv2d},
                        {"elementwise_add", op::elementwise_add},
                        {"elementwise_div", op::elementwise_div},
                        {"elementwise_max", op::elementwise_max},
                        {"elementwise_min", op::elementwise_min},
                        {"elementwise_mul", op::elementwise_mul},
                        {"elementwise_pow", op::elementwise_pow},
                        {"elementwise_sub", op::elementwise_sub},
                        {"relu", op::relu},
                        {"scale", op::scale},
                        {"split", op::split}};
            };

        } // namespace pdpd
    }     // namespace frontend
} // namespace ngraph
