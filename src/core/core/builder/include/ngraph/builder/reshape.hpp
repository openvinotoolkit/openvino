// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace builder
    {
        namespace opset1
        {
            /// \brief      Change shape of a value
            ///
            /// \param[in]  value  The value to be reshaped.
            /// \param[in]  shape  The new shape.
            ///
            /// \return     Reshape:v1 op.
            std::shared_ptr<Node> reshape(const Output<Node>& value, const Shape& shape);

            /// \brief Permute axes according to specified axes_order parameter.
            ///
            /// \param      The vlaue whose axes we want to permute.
            /// \param      axes_order The permutation of axes.
            ///
            /// \return     Transpose:v1 op.
            std::shared_ptr<Node> reorder_axes(const Output<Node>& value,
                                               std::vector<size_t> axes_order = {});

            /// \brief      Return transposed value (with axes in reversed order).
            ///
            /// \param      Value to transpose.
            ///
            /// \return     Transpose:v1 op.
            std::shared_ptr<Node> transpose(const Output<Node>& value);

            /// \brief       Flatten a value into a 2D matrix, with a static dividing axis.
            ///
            /// \param       The tensor to be flattened.
            /// \param       The axis dividing shape.
            ///
            /// \return      The new value will be a 2D matrix representing the flattened input
            /// node.
            std::shared_ptr<Node> flatten(const Output<Node>& value, int axis);

            /// \brief      Expands node tensor shape with empty axis at
            ///             specified position.
            ///
            /// \param[in]  value  The value to be expanded.
            /// \param[in]  axis   The position in the expanded axes where the
            ///                    new axis is placed.
            ///
            /// \return     Reshape:v1 op.
            std::shared_ptr<Node> expand_dims(const Output<Node>& value, std::size_t axis = 0);

            /// \brief      Remove empty axes from input tensor.
            ///
            /// \param[in]  value  The value to be squeezed.
            /// \param[in]  axes   The vector defining indexes of axes to be removed.
            ///
            /// \return     Reshape:v1 op.
            std::shared_ptr<Node> squeeze(const Output<Node>& value,
                                          std::vector<std::size_t> axes = {0});

            /// \brief      Collapse specified axes into single one.
            ///
            /// \note       Collapsed axes create a continuous range starting from outermost axis.
            ///
            /// \param[in]  value       The value to be reshaped.
            /// \param[in]  start_axis  The start axis index.
            /// \param[in]  end_axis    The end axis (inclusive) index.
            ///
            /// \return     The node with collapsed specified axes.
            ///
            std::shared_ptr<Node> collapse(const Output<Node>& value,
                                           const std::size_t start_axis,
                                           const std::size_t end_axis);
        } // namespace opset1
    }     // namespace  builder
} // namespace  ngraph
