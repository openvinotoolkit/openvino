// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace builder
    {
        /// \brief     Split value on specified axis into multiple parts.
        ///
        /// \param     value         The value to be split.
        /// \param     length_parts  The vector defining the lengths of each split part.
        /// \param     axis          The axis we split input node on. Default value is zero axis.
        ///
        /// \return     The vector containing multiple nodes we split input node into.
        ///
        NGRAPH_DEPRECATED("This builder was deprecated.")
        OutputVector split(const Output<Node>& value,
                           const std::vector<size_t>& length_parts,
                           int64_t axis = 0);

        /// \brief      Split node on specified axis into multiple parts.
        ///
        /// \param   value         The value to split.
        /// \param   split_parts   The number of parts we want to split output at given
        ///                        axis. The length of the axis to split must be divisible by
        ///                        this value.
        /// \param   axis          The axis we split input node on. Default value is zero axis.
        ///
        /// \note       This implementation supports negative `axis` values (similar to NumPy
        ///             indexing). This means that the axis to split on will be counted from
        ///             the back of the tensor (negative values are subtracted from its rank).
        ///
        /// \return     The vector containing multiple outputs we split input node into.
        ///
        NGRAPH_DEPRECATED("This builder was deprecated.")
        OutputVector split(const Output<Node>& value, size_t split_parts, int axis = 0);

        namespace opset1
        {
            /// \brief      Split value on specified axis into multiple parts.
            ///
            /// \param  value          The value to be split.
            /// \param  split_lengths  The vector defining the lengths of each split part.
            /// \param  axis           The axis we split input node on. Default value is zero
            ///                        axis.
            /// \note       This implementation supports negative `axis` values (similar to NumPy
            ///             indexing). This means that the axis to split on will be counted from
            ///             the back of the tensor (negative values are subtracted from its rank).
            ///
            /// \return     The vector containing multiple outputs we split input node into.
            ///             The vector is output of Split:v1 op
            ///
            OutputVector split(const Output<Node>& value,
                               const std::vector<size_t>& split_lengths,
                               int64_t axis = 0);

            /// \brief      Split value on specified axis into multiple parts.
            ///
            /// \param  value         The value to split.
            /// \param  num_splits    The number of parts we want to split output at given
            ///                       axis. The length of the axis to split must be divisible by
            ///                       this value.
            /// \param  axis          The axis we split input node on. Default value is zero
            ///                       axis.
            ///
            /// \note       This implementation supports negative `axis` values (similar to NumPy
            ///             indexing). This means that the axis to split on will be counted from
            ///             the back of the tensor (negative values are subtracted from its rank).
            ///
            /// \return     The vector containing multiple nodes we split input node into.
            ///             The vector is output of VariadicSplit:v1 op
            ///
            OutputVector split(const Output<Node>& value, size_t num_splits, int64_t axis = 0);
        } // namespace opset1
    }     // namespace builder
} // namespace ngraph
