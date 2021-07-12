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
                /// \brief Convert ONNX NonZero operation to an nGraph node.
                ///
                /// \param node   The ONNX node object representing this operation.
                ///
                /// \return The vector containing nGraph nodes producing output of ONNX NonZero
                ///         operation.
                OutputVector non_zero(const Node& node);

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
