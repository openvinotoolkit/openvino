// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector softmax(const Node& node);

            } // namespace set_1

            namespace set_11
            {
                OutputVector softmax(const Node& node);

            } // namespace set_11

            namespace set_13
            {
                OutputVector softmax(const Node& node);

            } // namespace set_13
        }     // namespace op

    } // namespace onnx_import

} // namespace ngraph
