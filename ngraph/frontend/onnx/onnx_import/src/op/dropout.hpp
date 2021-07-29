// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "onnx_import/core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_12
            {
                OutputVector dropout(const Node& node);
            } // namespace set_12

            namespace set_7
            {
                OutputVector dropout(const Node& node);
            } // namespace set_7

            namespace set_1
            {
                OutputVector dropout(const Node& node);
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
