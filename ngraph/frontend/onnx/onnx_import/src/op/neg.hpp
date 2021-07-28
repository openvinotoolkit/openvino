// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/negative.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline OutputVector neg(const Node& node) { return {-node.get_ng_inputs().at(0)}; }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
