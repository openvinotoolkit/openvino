// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace op
    {
        /// Root of all actual ops
        class NGRAPH_API Op : public Node
        {
        protected:
            Op()
                : Node()
            {
            }
            Op(const OutputVector& arguments);
        };
    } // namespace op
} // namespace ngraph
