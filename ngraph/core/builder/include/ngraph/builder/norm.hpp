// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

#include "ngraph/axis_set.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    namespace builder
    {
        /// \brief      Specifies method of bias application to avoid numerical problems
        enum class BiasMode
        {
            // Add bias to intermediate result
            ADD,
            // Calculate max of intermediate result and bias
            MAX
        };

        namespace opset1
        {
            /// \brief      Calculates L-0 norm of input tensor.
            ///
            /// \note       The L-0 norm represents the cardinality of elements different
            ///             from zero. This actually is not a "true" norm.
            ///
            /// \param[in]  value           The input tensor.
            /// \param[in]  reduction_axes  The axes along which we calculate norm.
            ///
            /// \return     L-0 norm of value. The output sub-graph is composed of v1 ops.
            ///
            std::shared_ptr<Node> l0_norm(const Output<Node>& value,
                                          const Output<Node>& reduction_axes);

            /// \brief      Calculates L-1 norm of a value.
            ///
            /// \note       The L-1 norm represents the sum of absolute values.
            ///
            /// \param[in]  value           The input tensor.
            /// \param[in]  reduction_axes  The axes along which we calculate norm.
            /// \param[in]  bias            The bias added to the calculated sum.
            ///
            /// \return     L-1 norm of value. The output sub-graph is composed of v1 ops.
            ///
            std::shared_ptr<Node> l1_norm(const Output<Node>& value,
                                          const Output<Node>& reduction_axes,
                                          float bias = 0.f);

            /// \brief      Calculates L-2 norm of input tensor.
            ///
            /// \note       The L-2 norm represents the square root of sum of squares of each
            ///             individual element.
            ///
            /// \param[in]  value           The input tensor.
            /// \param[in]  reduction_axes  The axes along which we calculate norm.
            /// \param[in]  bias            The bias combined with calculated sum.
            /// \param[in]  bias_mode       The method of bias application.
            /// \param[in]  keep_dims       The flag indicates if axes will be removed or kept.
            ///
            /// \return     L-2 norm of value. The output sub-graph is composed of v1 ops.
            ///
            std::shared_ptr<Node> l2_norm(const Output<Node>& value,
                                          const Output<Node>& reduction_axes,
                                          float bias = 0.f,
                                          BiasMode bias_mode = BiasMode::ADD,
                                          bool keep_dims = false);

            /// \brief      Creates node which calculates L-p norm on input tensor.
            ///
            /// \param[in]  value           The input tensor.
            /// \param[in]  reduction_axes  The axes along which we calculate norm.
            /// \param[in]  p_norm          The p norm to calculate.
            /// \param[in]  bias            The bias added to the calculated sum.
            ///
            /// \return     L-p norm of value. The output sub-graph is composed of v1 ops.
            ///
            std::shared_ptr<Node> lp_norm(const Output<Node>& value,
                                          const Output<Node>& reduction_axes,
                                          std::size_t p_norm = 2,
                                          float bias = 0.f);
        } // namespace opset1
    }     // namespace builder
} // namespace ngraph
