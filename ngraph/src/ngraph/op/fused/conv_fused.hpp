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

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Convolution + bias forward prop for batched convolution operation.
            class NGRAPH_API ConvolutionBias : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"ConvolutionBias", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ConvolutionBias() = default;
                ConvolutionBias(const std::shared_ptr<op::Convolution>& conv,
                                const Output<Node>& bias,
                                const bool with_relu = false);

                ConvolutionBias(const Output<Node>& data_batch,
                                const Output<Node>& filters,
                                const Output<Node>& bias,
                                const Strides& window_movement_strides,
                                const Strides& window_dilation_strides,
                                const CoordinateDiff& padding_below,
                                const CoordinateDiff& padding_above,
                                const Strides& data_dilation_strides,
                                const bool with_relu = false);

                ConvolutionBias(const Output<Node>& data_batch,
                                const Output<Node>& filters,
                                const Output<Node>& bias);

                const Strides& get_window_movement_strides() const
                {
                    return m_window_movement_strides;
                }
                const Strides& get_window_dilation_strides() const
                {
                    return m_window_dilation_strides;
                }
                const CoordinateDiff& get_padding_below() const { return m_padding_below; }
                const CoordinateDiff& get_padding_above() const { return m_padding_above; }
                const Strides& get_data_dilation_strides() const { return m_data_dilation_strides; }
                Output<Node> get_bias() { return input_value(2); }
                Output<Node> get_filters() { return input_value(1); }
                Output<Node> get_data_batch() { return input_value(0); }
                bool with_relu() const { return m_with_relu; }
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                virtual NodeVector decompose_op() const override;

                virtual void validate_and_infer_types() override;

            protected:
                Strides m_window_movement_strides;
                Strides m_window_dilation_strides;
                CoordinateDiff m_padding_below;
                CoordinateDiff m_padding_above;
                Strides m_data_dilation_strides;
                bool m_with_relu;
            };

            class NGRAPH_API ConvolutionBiasAdd : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"ConvolutionBiasAdd", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ConvolutionBiasAdd() = default;
                ConvolutionBiasAdd(const std::shared_ptr<op::v0::ConvolutionBias>& conv,
                                   const Output<Node>& sum_input,
                                   bool with_relu = false);

                ConvolutionBiasAdd(const Output<Node>& data_batch,
                                   const Output<Node>& filters,
                                   const Output<Node>& bias,
                                   const Output<Node>& sum_input,
                                   const Strides& window_movement_strides,
                                   const Strides& window_dilation_strides,
                                   const CoordinateDiff& padding_below,
                                   const CoordinateDiff& padding_above,
                                   const Strides& data_dilation_strides,
                                   bool with_relu = false);

                const Strides& get_window_movement_strides() const
                {
                    return m_window_movement_strides;
                }
                const Strides& get_window_dilation_strides() const
                {
                    return m_window_dilation_strides;
                }
                const CoordinateDiff& get_padding_below() const { return m_padding_below; }
                const CoordinateDiff& get_padding_above() const { return m_padding_above; }
                const Strides& get_data_dilation_strides() const { return m_data_dilation_strides; }
                Output<Node> get_filters() { return input_value(1); }
                Output<Node> get_data_batch() { return input_value(0); }
                bool with_relu() const { return m_with_relu; }
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                virtual NodeVector decompose_op() const override;

                virtual void validate_and_infer_types() override;

            protected:
                Strides m_window_movement_strides;
                Strides m_window_dilation_strides;
                CoordinateDiff m_padding_below;
                CoordinateDiff m_padding_above;
                Strides m_data_dilation_strides;
                bool m_with_relu;
            };
        }
        using v0::ConvolutionBias;
        using v0::ConvolutionBiasAdd;
    }
}
