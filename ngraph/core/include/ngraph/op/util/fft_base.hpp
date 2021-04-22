// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            /// \brief Base class for operations DFT and DFT.
            class NGRAPH_API FFTBase : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                FFTBase() = default;

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

            protected:
                /// \brief Constructs an FFT operation. FFT is performed for full size axes.
                ///
                /// \param data  Input data
                /// \param axes Axes to perform FFT
                FFTBase(const Output<Node>& data, const Output<Node>& axes);

                /// \brief Constructs a FFT operation.
                ///
                /// \param data  Input data
                /// \param axes Axes to perform FFT
                /// \param signal_size Signal sizes for 'axes'
                FFTBase(const Output<Node>& data,
                        const Output<Node>& axes,
                        const Output<Node>& signal_size);

                void validate();
            };
        } // namespace util
    }     // namespace op
} // namespace ngraph
