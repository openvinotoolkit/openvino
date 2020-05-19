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

#include <memory>

#include "ngraph/deprecated.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Batchnorm for training operation
            class NGRAPH_API BatchNormTraining : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"BatchNormTraining", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                BatchNormTraining() = default;
                /// \param input Must have rank >= 2, [., C, ...]
                /// \param gamma gamma scaling for normalized value. [C]
                /// \param beta bias added to the scaled normalized value [C]
                /// \param epsilon Avoids divsion by 0 if input has 0 variance
                BatchNormTraining(const Output<Node>& input,
                                  const Output<Node>& gamma,
                                  const Output<Node>& beta,
                                  double epsilon);

                bool visit_attributes(AttributeVisitor& visitor) override;

                NGRAPH_DEPRECATED_DOC
                /// In this version of BatchNorm:
                ///
                /// MEAN AND VARIANCE: computed directly from the content of 'input'.
                ///
                /// OUTPUT VALUE: A tuple with the following structure:
                ///   [0] - The normalization of 'input'.
                ///   [1] - The per-channel means of (pre-normalized) 'input'.
                ///   [2] - The per-channel variances of (pre-normalized) 'input'.
                ///
                /// AUTODIFF SUPPORT: yes: 'generate_adjoints(...)' works as expected.
                ///
                /// SHAPE DETAILS:
                ///   gamma:     must have rank 1, with the same span as input's channel axis.
                ///   beta:      must have rank 1, with the same span as input's channel axis.
                ///   input:     must have rank >= 2.  The second dimension represents the channel
                ///   axis
                ///              and must have a span of at least 1.
                ///   output[0]: shall have the same shape as 'input'.
                ///   output[1]: shall have rank 1, with the same span as input's channel axis.
                ///   output[2]: shall have rank 1, with the same span as input's channel axis.
                NGRAPH_DEPRECATED("Use another constructor")
                BatchNormTraining(double eps,
                                  const Output<Node>& gamma,
                                  const Output<Node>& beta,
                                  const Output<Node>& input);

                void validate_and_infer_types() override;

                double get_eps_value() const { return m_epsilon; }
                void set_eps_value(double epsilon) { m_epsilon = epsilon; }
                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;

                static constexpr size_t INPUT_GAMMA = 0;
                static constexpr size_t INPUT_BETA = 1;
                static constexpr size_t INPUT_DATA = 2;

            private:
                double m_epsilon;
            };

            class NGRAPH_API BatchNormInference : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"BatchNormInference", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                BatchNormInference() = default;
                /// \param input [., C, ...]
                /// \param gamma gamma scaling for normalized value. [C]
                /// \param beta bias added to the scaled normalized value [C]
                /// \param mean value for mean normalization [C]
                /// \param variance value for variance normalization [C]
                /// \param epsilon Avoids divsion by 0 if input has 0 variance
                BatchNormInference(const Output<Node>& input,
                                   const Output<Node>& gamma,
                                   const Output<Node>& beta,
                                   const Output<Node>& mean,
                                   const Output<Node>& variance,
                                   double epsilon);

                bool visit_attributes(AttributeVisitor& visitor) override;

                NGRAPH_DEPRECATED_DOC
                /// In this version of BatchNorm:
                ///
                /// MEAN AND VARIANCE: provided by the 'mean' and 'variance' parameters.
                ///
                /// OUTPUT VALUE: a single tensor with the normalized value of 'input'.
                ///
                /// AUTODIFF SUPPORT:
                ///   - 'generate_adjoints(...) may throw an exception.
                ///
                /// SHAPE DETAILS:
                ///   gamma:    must have rank 1, with the same span as input's channel axis.
                ///   beta:     must have rank 1, with the same span as input's channel axis.
                ///   input:    must have rank >= 2. The second dimension represents the channel
                ///   axis
                ///             and must have a span of at least 1.
                ///   mean:     must have rank 1, with the same span as input's channel axis.
                ///   variance: must have rank 1, with the same span as input's channel axis.
                ///   output:   shall have the same shape as 'input'.
                NGRAPH_DEPRECATED("Use another constructor")
                BatchNormInference(double eps,
                                   const Output<Node>& gamma,
                                   const Output<Node>& beta,
                                   const Output<Node>& input,
                                   const Output<Node>& mean,
                                   const Output<Node>& variance);

                void validate_and_infer_types() override;

                double get_eps_value() const { return m_epsilon; }
                void set_eps_value(double epsilon) { m_epsilon = epsilon; }
                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                               const OutputVector& /* deltas */) override
                {
                    throw ngraph_error("Invalid operation");
                }

            private:
                static constexpr size_t INPUT_GAMMA = 0;
                static constexpr size_t INPUT_BETA = 1;
                static constexpr size_t INPUT_DATA = 2;
                static constexpr size_t INPUT_MEAN = 3;
                static constexpr size_t INPUT_VARIANCE = 4;

                double m_epsilon;
            };

            class NGRAPH_API BatchNormTrainingBackprop : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"BatchNormTrainingBackprop", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                BatchNormTrainingBackprop() = default;
                BatchNormTrainingBackprop(const Output<Node>& input,
                                          const Output<Node>& gamma,
                                          const Output<Node>& beta,
                                          const Output<Node>& mean,
                                          const Output<Node>& variance,
                                          const Output<Node>& delta,
                                          double epsilon);

                bool visit_attributes(AttributeVisitor& visitor) override;

                NGRAPH_DEPRECATED_DOC
                NGRAPH_DEPRECATED("Use another constructor")
                BatchNormTrainingBackprop(double epsilon,
                                          const Output<Node>& gamma,
                                          const Output<Node>& beta,
                                          const Output<Node>& input,
                                          const Output<Node>& mean,
                                          const Output<Node>& variance,
                                          const Output<Node>& delta);

                void validate_and_infer_types() override;

                double get_eps_value() const { return m_epsilon; }
                void set_eps_value(double epsilon) { m_epsilon = epsilon; }
                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

            private:
                static constexpr size_t INPUT_GAMMA = 0;
                static constexpr size_t INPUT_BETA = 1;
                static constexpr size_t INPUT_DATA = 2;
                static constexpr size_t INPUT_MEAN = 3;
                static constexpr size_t INPUT_VARIANCE = 4;
                static constexpr size_t INPUT_DELTA = 5;

                double m_epsilon;
            };
        }
        using v0::BatchNormInference;
        using v0::BatchNormTraining;
        using v0::BatchNormTrainingBackprop;
    }
}
