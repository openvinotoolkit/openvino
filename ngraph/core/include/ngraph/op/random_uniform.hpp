// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v8
        {
            /// \brief Tensor RandomUniform operation.
            class NGRAPH_API RandomUniform : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                RandomUniform() = default;

                ///
                /// \brief      Constructs a RandomUniform operation.
                ///
                /// \param      out_shape         Node producing the tensor with output shape.
                /// \param      min_val           Node producing the tensor with minimum value.
                /// \param      max_val           Node producing the tensor with maximum value.
                /// \param      out_type          Output type of the tensor.
                /// \param      seed              Global seed value.
                /// \param      seed2             Operational seed value.
                RandomUniform(const Output<Node>& out_shape,
                              const Output<Node>& min_val,
                              const Output<Node>& max_val,
                              const ngraph::element::Type& out_type,
                              int64_t seed,
                              int64_t seed2);

                void validate_and_infer_types() override;

                bool visit_attributes(AttributeVisitor& visitor) override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The output tensor type.
                const ngraph::element::Type& get_out_type() const { return m_output_type; }
                void set_out_type(const ngraph::element::Type& output_type)
                {
                    m_output_type = output_type;
                }

                /// \return The global seed value.
                const uint64_t& get_seed() const { return m_seed; }
                void set_seed(const uint64_t& seed) { m_seed = seed; }

                /// \return The operational seed value.
                const uint64_t& get_seed2() const { return m_seed2; }
                void set_seed2(const uint64_t& seed2) { m_seed2 = seed2; }

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;

            protected:
                ngraph::element::Type m_output_type;
                uint64_t m_seed;
                uint64_t m_seed2;
            };
        } // namespace v8
    }     // namespace op
} // namespace ngraph
