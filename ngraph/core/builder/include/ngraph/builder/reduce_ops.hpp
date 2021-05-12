// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/axis_set.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    namespace builder
    {
        namespace opset1
        {
            // clang-format off
            /// \brief Sum-based Mean of a Tensor.
            ///
            /// Calculates
            ///
            /// \f$\sum_{i=1}^{N} \frac{x_i}{N}\f$
            ///
            /// Where `i` traverses all of the axes provided in `reduction_axes`
            ///
            /// ## Inputs
            ///
            /// |                  | Type                              | Description |                                          |
            /// | ---------------- | --------------------------------- | -------------------------------------------------------|
            /// | `node`           | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | An input tensor of any shape                           |
            /// | `reduction_axes` | AxesSet                           | The axes to eliminate through reduction (0 indexed).   |
            /// | `keep_dims`      | bool                              | If set to true it holds reduced axes.                  |
            ///
            /// ## Output
            ///
            /// | Type                                      | Description                                                                                                      |
            /// | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
            /// | \f$E[\textit{delete}(A,d_1,\dots,d_n)]\f$ | The tensor \f$T\f$, where \f$T\f$ is the input tensor with the `reduction_axes` \f$A\f$ eliminated by reduction. |
            // clang-format on
            std::shared_ptr<Node> mean(const Output<Node>& node,
                                       const AxisSet& reduction_axes,
                                       bool keep_dims = false);

            std::shared_ptr<Node> mean(const Output<Node>& node,
                                       const Output<Node>& reduction_axes,
                                       bool keep_dims = false);

            // clang-format off
            /// \brief Sum-based Variance of a Tensor.
            ///
            /// If bessel_correct is true, calculates
            ///
            /// \f$\frac{\sum_{i=1}^{N}\left(x_i-\bar{x}\right)^2}{N-1}\f$
            ///
            /// else, calculates
            ///
            /// \f$\frac{\sum_{i=1}^{N}\left(x_i-\bar{x}\right)^2}{N}\f$
            ///
            /// Where `i` traverses all of the axes provided in `reduction_axes` and \f$\bar{x} = \sum_{i=1}^{N} \frac{x_i}{N}\f$
            ///
            /// ## Inputs
            ///
            /// |                     | Type                              | Description                                                  |
            /// | ------------------- | --------------------------------- | ------------------------------------------------------------ |
            /// | `value              | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | An input tensor of any shape                                 |
            /// | `reduction_axes`    | AxesSet                           | The axes to eliminate through reduction (0 indexed).         |
            /// | `bessel_correction` | bool (default = false)            | Enable Bessel's correction to std_dev for Small sample sizes |
            ///
            /// ## Output
            ///
            /// | Type                                      | Description                                                                                                      |
            /// | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
            /// | \f$E[\textit{delete}(A,d_1,\dots,d_n)]\f$ | The tensor \f$T\f$, where \f$T\f$ is the input tensor with the `reduction_axes` \f$A\f$ eliminated by reduction. |
            // clang-format on
            std::shared_ptr<Node> variance(const Output<Node>& value,
                                           const AxisSet& reduction_axes,
                                           const bool bessel_correction = false);

            std::shared_ptr<Node> variance(const Output<Node>& value,
                                           const Output<Node>& reduction_axes,
                                           bool keep_dims = false,
                                           bool bessel_correction = false);
        } // namespace opset1

    } // namespace builder
} // namespace ngraph
