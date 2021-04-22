// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            // clang-format off
        /// \brief Elementwise Local Response Normalization (LRN) operation.
        ///
        /// ## Inputs
        ///
        /// |       | Type                                    | Description                                     |
        /// | ----- | --------------------------------------- | ----------------------------------------------- |
        /// | `arg` | \f$N[n, c, d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and numeric element type. |
        ///
        /// ## Output
        ///
        /// | Type                         | Description                                                                                                                                                                                  |
        /// | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        /// | \f$N[n, c, d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[n, c, d_1,\dots,d_n] = \frac{N[n,i,d_1,\dots,d_n]}{ (bias + alpha * (\sum_{i=max(0,(nsize-1)/2)}^{min(C, (nsize-1)/2)+1} N[n,i,d_1,\dots,d_n]^{2}) ^ {2})}\f$ |
            // clang-format on
            class NGRAPH_API LRN : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"LRN", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a LRN operation.
                LRN() = default;
                /// \brief Constructs a LRN operation.
                ///
                /// \param arg Node that produces the input tensor.
                LRN(const Output<Node>& arg, double alpha, double beta, double bias, size_t size);

                LRN(const Output<Node>& arg,
                    const Output<Node>& axes,
                    double alpha,
                    double beta,
                    double bias,
                    size_t size);

                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                void validate_and_infer_types() override;

                double get_alpha() const { return m_alpha; }
                void set_alpha(double alpha) { m_alpha = alpha; }
                double get_beta() const { return m_beta; }
                void set_beta(double beta) { m_beta = beta; }
                double get_bias() const { return m_bias; }
                void set_bias(double bias) { m_bias = bias; }
                size_t get_nsize() const { return m_size; }
                void set_nsize(size_t size) { m_size = size; }
                AxisSet get_reduction_axes() const;

            protected:
                double m_alpha;
                double m_beta;
                double m_bias;
                size_t m_size;
            };
        } // namespace v0
        using v0::LRN;
    } // namespace op
} // namespace ngraph
