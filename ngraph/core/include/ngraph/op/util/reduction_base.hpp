// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            class NGRAPH_API ReductionBase : public Op
            {
            protected:
                /// \brief Constructs a reduction operation.
                ReductionBase();

                /// \brief Constructs a reduction operation.
                ///
                /// \param arg Output that produces the first input tensor.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                ReductionBase(const Output<Node>& arg, const Output<Node>& reduction_axes);

                /// \brief      Infers reduction operations output shape.
                ///
                /// \param[in] keep_dims    Reduction operation keeps dimensions.
                ///
                /// \return Partial shape of the output.
                PartialShape infer_reduction_output_shape(const bool keep_dims);

            public:
                NGRAPH_RTTI_DECLARATION;
            };
        } // namespace util
    }     // namespace op
} // namespace ngraph