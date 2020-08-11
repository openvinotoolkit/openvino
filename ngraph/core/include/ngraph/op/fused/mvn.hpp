//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Operator performing Mean Variance Normalization
            ///
            class NGRAPH_API MVN : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"MVN", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                MVN() = default;
                /// \brief Constructs an MVN operation.
                ///
                /// \param data Input tensor with data
                /// \param normalize_variance flag that denotes whether to perform variance
                ///                           normalization.
                /// \param across_channels flag that denotes if mean values are shared across
                /// channels.
                /// \param eps the number to be added to the variance to avoid division by zero when
                ///            normalizing the value
                ///
                MVN(const Output<Node>& data,
                    bool across_channels = true,
                    bool normalize_variance = true,
                    double eps = 1e-9);

                /// \brief Constructs an MVN operation.
                ///
                /// \param data Input tensor with data
                /// \param reduction_axes A list of axes, along which to reduce.
                /// \param normalize_variance flag that denotes whether to perform variance
                ///                           normalization.
                /// \param eps the number to be added to the variance to avoid division by zero when
                ///            normalizing the value
                ///
                MVN(const Output<Node>& data,
                    AxisSet reduction_axes,
                    bool normalize_variance = true,
                    double eps = 1e-9);

                virtual OutputVector decompose_op() const override;

                virtual void validate_and_infer_types() override;

                virtual bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                double get_eps() const { return m_eps; }
                bool get_across_channels() const { return m_across_channels; }
                bool get_normalize_variance() const { return m_normalize_variance; }
                AxisSet get_reduction_axes() const { return m_reduction_axes; }
                void set_reduction_axes(AxisSet axes) { m_reduction_axes = axes; }
            private:
                double m_eps = 1e-9;
                bool m_across_channels;
                bool m_normalize_variance;
                AxisSet m_reduction_axes;
            };
        }
        using v0::MVN;
    } // namespace op
} // namespace ngraph
