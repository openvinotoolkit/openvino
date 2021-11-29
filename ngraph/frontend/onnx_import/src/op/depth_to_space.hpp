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
                /// \brief      Permutes input tensor data from depth into blocks of spatial data.
                ///
                /// \note       Values from the depth dimension (assuming NCHW layout) are moved in
                ///             spatial blocks to the height and width dimensions.
                ///
                /// \param[in]  node  The ONNX input node describing operation.
                ///
                /// \return     OutputVector containing Tensor with shape:
                ///             [N, C/(blocksize * blocksize), H * blocksize, W * blocksize]
                OutputVector depth_to_space(const Node& node);
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
