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
                /// \brief ONNX Shrink operator
                ///
                /// \note It operates on a single input tensor and two attributes: lambd and bias.
                ///       Input values greater or equal to '-lambd' and less or equal to 'lambd' are
                ///       zeroed-out. 'Bias' is added to the values that are less than '-lambd'
                ///       and subtracted from values greater than 'lambd'.
                OutputVector shrink(const Node& node);
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
