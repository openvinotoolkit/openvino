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

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Relu(Convolution) forward prop for batched convolution operation.
            class NGRAPH_API QuantizedConvolutionRelu : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"QuantizedConvolutionRelu", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                QuantizedConvolutionRelu() = default;
                QuantizedConvolutionRelu(const Output<Node>& data_batch,
                                         const Output<Node>& filters,
                                         const Strides& window_movement_strides,
                                         const Strides& window_dilation_strides,
                                         const CoordinateDiff& padding_below,
                                         const CoordinateDiff& padding_above,
                                         const Strides& data_dilation_strides,
                                         const Output<Node>& scale);

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
                bool with_relu() const { return true; }
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

            protected:
                Strides m_window_movement_strides;
                Strides m_window_dilation_strides;
                CoordinateDiff m_padding_below;
                CoordinateDiff m_padding_above;
                Strides m_data_dilation_strides;
            };
        }
        using v0::QuantizedConvolutionRelu;
    }
}
