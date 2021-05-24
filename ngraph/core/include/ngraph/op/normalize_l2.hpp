// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/fused_op.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief  Normalization input tensor with L2 norm.
            ///
            class NGRAPH_API NormalizeL2 : public ngraph::op::util::FusedOp
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                NormalizeL2();
                ///
                /// \brief      Constructs a Normalize operation.
                ///
                /// \param      data            - Node producing the input tensor
                /// \param      axes            - Node indicating axes along which reduction is
                ///                               calculated
                /// \param      eps             - The epsilon added to L2 norm.
                /// \param      eps_mode        - Specifies how eps is combined with L2 value
                /// calculated
                ///                               before division
                ///
                NormalizeL2(const Output<Node>& data,
                            const Output<Node>& axes,
                            float eps,
                            EpsMode eps_mode);

                bool visit_attributes(AttributeVisitor& visitor) override;
                float get_eps() const { return m_eps; }
                EpsMode get_eps_mode() const { return m_eps_mode; }
                virtual OutputVector decompose_op() const override;
                virtual void pre_validate_and_infer_types() override;
                AxisSet get_reduction_axes() const;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

            protected:
                float m_eps;
                EpsMode m_eps_mode;
            };
        } // namespace v0
        using v0::NormalizeL2;
    } // namespace op
} // namespace ngraph

NGRAPH_SUPPRESS_DEPRECATED_END
