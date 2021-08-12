// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace reshape
        {
            /// \brief      Infer `output_shape` dimension values.
            ///
            /// \par Inference rules
            ///     \li         The input_shape may consist at most on -1 value. In this case the
            ///                 value is inferred from the size of the tensor and the remaining
            ///                 dimensions.
            ///     \li         If a dimension value is equal to 0, then its output value is going
            ///                 to be copied from the input_shape argument.
            ///
            /// \param[in]  node_name     The node name.
            /// \param[in]  input_shape   The input node shape.
            /// \param[in]  output_shape  The requested output shape for the input node data.
            ///
            /// \return     A vector containing new, valid node shape.
            ///
            std::vector<std::size_t> infer_dimensions(const std::string& node_name,
                                                      const std::vector<std::size_t>& input_shape,
                                                      const std::vector<std::size_t>& output_shape);

            /// \brief      Handle a node which represents a scalar value.
            ///
            /// \note       Some ONNX nodes, which should provide scalar values are given as
            ///             tensors of shape {1}. This function will provide a reshape of
            ///             such a node with Shape{1} into a scalar with Shape{}.
            ///
            /// \param[in]  node   Node to reshape.
            ///
            /// \return     Original node or a node representing a reshape of the original.
            ///
            Output<ngraph::Node> interpret_as_scalar(const Output<ngraph::Node>& node);

            /// \brief      Reshape node from shape {C} to {1, C, 1, 1,...}
            ///
            /// \note       This function will reshape the input node
            ///             with a shape of {C} into a node with Shape{1, C, 1, 1, ..}.
            ///             The most common input to this function would be scale or bias to
            ///             BatchNorm or bias to Conv.
            ///
            /// \param[in]  node            Node to reshape.
            /// \param[in]  expected_rank   Expected rank size
            ///
            /// \return     Original node or a node representing a reshape of the original.
            ///
            Output<ngraph::Node>
                reshape_channel_shaped_node_to_nchw(const Output<ngraph::Node>& node,
                                                    const Output<ngraph::Node>& expected_rank);

        } // namespace  reshape
    }     // namespace onnx_import
} // namespace ngraph
