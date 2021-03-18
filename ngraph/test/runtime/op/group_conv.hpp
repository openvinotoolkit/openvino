//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "backend_visibility.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/fused_op.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Group Convolution
            class BACKEND_API GroupConvolution : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"GroupConvolution", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                GroupConvolution() = default;
                GroupConvolution(const Output<Node>& data_batch,
                                 const Output<Node>& filters,
                                 const Strides& window_movement_strides,
                                 const Strides& window_dilation_strides,
                                 const CoordinateDiff& padding_below,
                                 const CoordinateDiff& padding_above,
                                 const Strides& data_dilation_strides,
                                 const size_t groups,
                                 const PadType& pad_type = PadType::EXPLICIT);

                // constructor which accept groups included in filters shape.
                GroupConvolution(const Output<Node>& data_batch,
                                 const Output<Node>& filters,
                                 const Strides& window_movement_strides,
                                 const Strides& window_dilation_strides,
                                 const CoordinateDiff& padding_below,
                                 const CoordinateDiff& padding_above,
                                 const Strides& data_dilation_strides,
                                 const PadType& pad_type = PadType::EXPLICIT);
                Shape get_weights_dimensions() const;
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
                size_t get_groups() const { return m_groups; };
                const PadType& get_pad_type() const { return m_pad_type; }
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                virtual OutputVector decompose_op() const override;

                virtual void pre_validate_and_infer_types() override;
                virtual void post_validate_and_infer_types() override;

                bool has_groups_in_filters() const { return m_groups_in_filters; }

            protected:
                Strides m_window_movement_strides;
                Strides m_window_dilation_strides;
                CoordinateDiff m_padding_below;
                CoordinateDiff m_padding_above;
                Strides m_data_dilation_strides;
                size_t m_groups;
                PadType m_pad_type{PadType::NOTSET};

            private:
                bool m_groups_in_filters;
            };

            /// \brief Group Convolution data batch backprop
            class BACKEND_API GroupConvolutionBackpropData : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"GroupConvolutionBackpropData", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                GroupConvolutionBackpropData() = default;
                GroupConvolutionBackpropData(const Output<Node>& data_batch,
                                             const Output<Node>& filters,
                                             const Output<Node>& output_delta,
                                             const Strides& window_movement_strides,
                                             const Strides& window_dilation_strides,
                                             const CoordinateDiff& padding_below,
                                             const CoordinateDiff& padding_above,
                                             const size_t groups);

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
                size_t get_groups() const { return m_groups; };
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                virtual OutputVector decompose_op() const override;

                virtual void pre_validate_and_infer_types() override;

            protected:
                Strides m_window_movement_strides;
                Strides m_window_dilation_strides;
                CoordinateDiff m_padding_below;
                CoordinateDiff m_padding_above;
                size_t m_groups;
            };
        }
    } // namespace op
} // namespace ngraph

NGRAPH_SUPPRESS_DEPRECATED_END
