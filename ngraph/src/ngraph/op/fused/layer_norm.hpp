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
            /// \brief Layer Normalization
            ///
            class NGRAPH_API LayerNorm : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"LayerNorm", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                LayerNorm() = default;
                /// \brief Constructs an LayerNorm operation.
                ///
                /// \param data Input tensor
                /// \param scale Scale tensor
                /// \param bias Bias tensor
                /// \param keep_stats Generated addition output mean and variance, default true
                /// \param begin_norm_axis Axis where normalization starts, default - -1
                /// \param epsilon Small number to add for stability of rsqrt, default 1e-5
                LayerNorm(const Output<Node>& data,
                          const Output<Node>& scale,
                          const Output<Node>& bias,
                          bool keep_stats = true,
                          int64_t begin_norm_axis = 1,
                          double epsilon = 1e-5);

                LayerNorm(const Output<Node>& data,
                          bool keep_stats = true,
                          int64_t begin_norm_axis = 1,
                          double epsilon = 1e-5);

                virtual NodeVector decompose_op() const override;

                void pre_validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool get_keep_stats() const { return m_keep_stats; }
                bool get_use_affine() const { return m_use_affine; }
                double get_epsilon() const { return m_epsilon; }
                int64_t get_begin_norm_axis() const { return m_begin_norm_axis; }
            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;

            private:
                bool m_keep_stats{true};
                bool m_use_affine{true};
                int64_t m_begin_norm_axis{1};
                double m_epsilon{1e-5};
            };

            /// \brief Layer Normalization Backprop
            ///
            class NGRAPH_API LayerNormBackprop : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"LayerNormBackprop", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                LayerNormBackprop() = default;
                /// \brief Constructs an LayerNormBackprop operation.
                ///
                /// \param data Input tensor
                /// \param mean Mean tensor from fprop
                /// \param variance Variance tensor from fprop
                /// \param delta Delta tensor
                /// \param scale Scale tensor
                /// \param begin_norm_axis Axis where normalization starts, default - -1
                /// \param epsilon Small number to add for stability of rsqrt, default 1e-5
                LayerNormBackprop(const Output<Node>& data,
                                  const Output<Node>& delta,
                                  const Output<Node>& mean,
                                  const Output<Node>& variance,
                                  const Output<Node>& scale,
                                  int64_t begin_norm_axis = 1,
                                  double epsilon = 1e-5);

                LayerNormBackprop(const Output<Node>& data,
                                  const Output<Node>& delta,
                                  const Output<Node>& mean,
                                  const Output<Node>& variance,
                                  int64_t begin_norm_axis = 1,
                                  double epsilon = 1e-5);

                LayerNormBackprop(const Output<Node>& data,
                                  const Output<Node>& delta,
                                  const Output<Node>& scale,
                                  int64_t begin_norm_axis = 1,
                                  double epsilon = 1e-5);

                LayerNormBackprop(const Output<Node>& data,
                                  const Output<Node>& delta,
                                  int64_t begin_norm_axis = 1,
                                  double epsilon = 1e-5);

                virtual NodeVector decompose_op() const override;

                void pre_validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool get_use_stats() const { return m_use_stats; }
                bool get_use_affine() const { return m_use_affine; }
                double get_epsilon() const { return m_epsilon; }
                int64_t get_begin_norm_axis() const { return m_begin_norm_axis; }
            private:
                bool m_use_stats{true};
                bool m_use_affine{true};
                int64_t m_begin_norm_axis{1};
                double m_epsilon{1e-5};
            };
        }
        using v0::LayerNorm;
        using v0::LayerNormBackprop;
    }
}
