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

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Batched average pooling operation, with optional padding and window stride.
            ///
            class NGRAPH_API AvgPool : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"AvgPool", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a batched average pooling operation.
                AvgPool() = default;

                /// \brief Constructs a batched average pooling operation.
                ///
                /// \param arg The output producing the input data batch tensor.<br>
                /// `[d1, dn]`
                /// \param window_shape The window shape.<br>
                /// `[n]`
                /// \param window_movement_strides The window movement strides.<br>
                /// `[n]`
                /// \param padding_below The below-padding shape.<br>
                /// `[n]`
                /// \param padding_above The above-padding shape.<br>
                /// `[n]`
                /// \param include_padding_in_avg_computation If true then averages include padding
                /// elements, each treated as the number zero.  If false, padding elements are
                /// entirely ignored when computing averages. \param pad_type Padding type to use
                /// for additional padded dimensions \param ceil_mode Whether to use ceiling while
                /// computing output shape.
                AvgPool(const Output<Node>& arg,
                        const Shape& window_shape,
                        const Strides& window_movement_strides,
                        const Shape& padding_below,
                        const Shape& padding_above,
                        bool include_padding_in_avg_computation,
                        const PadType& pad_type,
                        bool ceil_mode);

                /// \brief Constructs a batched average pooling operation.
                ///
                /// \param arg The output producing the input data batch tensor.<br>
                /// `[d1, dn]`
                /// \param window_shape The window shape.<br>
                /// `[n]`
                /// \param window_movement_strides The window movement strides.<br>
                /// `[n]`
                /// \param padding_below The below-padding shape.<br>
                /// `[n]`
                /// \param padding_above The above-padding shape.<br>
                /// `[n]`
                /// \param include_padding_in_avg_computation If true then averages include padding
                /// elements, each treated as the number zero.  If false, padding elements are
                /// entirely ignored when computing averages. \param pad_type Padding type to use
                /// for additional padded dimensions
                AvgPool(const Output<Node>& arg,
                        const Shape& window_shape,
                        const Strides& window_movement_strides,
                        const Shape& padding_below,
                        const Shape& padding_above,
                        bool include_padding_in_avg_computation,
                        const PadType& pad_type);

                /// \brief Constructs a batched average pooling operation.
                ///
                /// \param arg The output producing the input data batch tensor.<br>
                /// `[d1, dn]`
                /// \param window_shape The window shape.<br>
                /// `[n]`
                /// \param window_movement_strides The window movement strides.<br>
                /// `[n]`
                /// \param padding_below The below-padding shape.<br>
                /// `[n]`
                /// \param padding_above The above-padding shape.<br>
                /// `[n]`
                /// \param include_padding_in_avg_computation If true then averages include padding
                /// elements, each treated as the number zero.  If false, padding elements are
                /// entirely ignored when computing averages.
                AvgPool(const Output<Node>& arg,
                        const Shape& window_shape,
                        const Strides& window_movement_strides,
                        const Shape& padding_below,
                        const Shape& padding_above,
                        bool include_padding_in_avg_computation = false);

                /// \brief Constructs a batched, unpadded average pooling operation (i.e., all
                /// padding shapes are set to 0).
                ///
                /// \param arg The output producing the input data batch tensor.<br>
                /// `[d1, ..., dn]`
                /// \param window_shape The window shape.<br>
                /// `[n]`
                /// \param window_movement_strides The window movement strides.<br>
                /// `[n]`
                AvgPool(const Output<Node>& arg,
                        const Shape& window_shape,
                        const Strides& window_movement_strides);

                /// \brief Constructs an unstrided batched convolution operation (i.e., all window
                /// movement strides are 1 and all padding shapes are set to 0).
                ///
                /// \param arg The output producing the input data batch tensor.<br>
                /// `[d1, ..., dn]`
                /// \param window_shape The window shape.<br>
                /// `[n]`
                AvgPool(const Output<Node>& arg, const Shape& window_shape);

                bool visit_attributes(AttributeVisitor& visitor) override;

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;

                /// \return The window shape.
                const Shape& get_window_shape() const;
                void set_window_shape(const Shape& window_shape);
                /// \return The window movement strides.
                const Strides& get_window_movement_strides() const;
                void set_window_movement_strides(const Strides& window_movement_strides);
                /// \return The below-padding shape.
                const Shape& get_padding_below() const;
                void set_padding_below(const Shape& padding_below);
                /// \return The above-padding shape.
                const Shape& get_padding_above() const;
                void set_padding_above(const Shape& padding_above);
                bool get_include_padding_in_avg_computation() const;
                void
                    set_include_padding_in_avg_computation(bool include_padding_in_avg_computation);
                /// \return The pad type for pooling.
                const PadType& get_pad_type() const;
                void set_pad_type(const PadType& pad_type);
                bool get_ceil_mode() const;
                void set_ceil_mode(bool ceil_mode);
                /// \return The default value for AvgPool.
                virtual std::shared_ptr<Node> get_default_value() const override;

            protected:
                Shape m_window_shape;
                Strides m_window_movement_strides;
                Shape m_padding_below;
                Shape m_padding_above;
                bool m_include_padding_in_avg_computation{false};
                PadType m_pad_type{PadType::EXPLICIT};
                bool m_ceil_mode{false};
            };

            class NGRAPH_API AvgPoolBackprop : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"AvgPoolBackprop", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                AvgPoolBackprop() = default;
                AvgPoolBackprop(const Shape& forward_arg_shape,
                                const Output<Node>& delta,
                                const Shape& window_shape,
                                const Strides& window_movement_strides,
                                const Shape& padding_below,
                                const Shape& padding_above,
                                bool include_padding_in_avg_computation);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                const Shape& get_forward_arg_shape() const;
                void set_forward_arg_shape(const Shape& forward_arg_shape);
                const Shape& get_window_shape() const;
                void set_window_shape(const Shape& window_shape);
                const Strides& get_window_movement_strides() const;
                void set_window_movement_strides(const Strides& window_movement_strides);
                const Shape& get_padding_below() const;
                void set_padding_below(const Shape& padding_below);
                const Shape& get_padding_above() const;
                void set_padding_above(const Shape& padding_abve);
                bool get_include_padding_in_avg_computation() const;
                void
                    set_include_padding_in_avg_computation(bool include_padding_in_avg_computation);

            protected:
                Shape m_forward_arg_shape;
                Shape m_window_shape;
                Strides m_window_movement_strides;
                Shape m_padding_below;
                Shape m_padding_above;
                bool m_include_padding_in_avg_computation{false};
            };
        } // namespace v0

        namespace v1
        {
            /// \brief Batched average pooling operation.
            ///
            class NGRAPH_API AvgPool : public Op
            {
            public:
                RTTI_DECLARATION;
                //static constexpr NodeTypeInfo type_info{"AvgPool", 1};
                //const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a batched average pooling operation.
                AvgPool() = default;

                ///
                /// \brief      Constructs a batched average pooling operation.
                ///
                /// \param      arg            The output producing the input data batch tensor.<br>
                ///                            `[d1, dn]`
                /// \param      strides        The strides.<br> `[n]`
                /// \param      pads_begin     The beginning of padding shape.<br> `[n]`
                /// \param      pads_end       The end of padding shape.<br> `[n]`
                /// \param      kernel         The kernel shape.<br> `[n]`
                /// \param      exclude_pad    If false then averages include padding elements, each
                ///                            treated as the number zero.  If true, padding
                ///                            elements
                ///                            are entirely ignored when computing averages.
                /// \param      rounding_type  Whether to use ceiling or floor rounding type while
                ///                            computing output shape.
                /// \param      auto_pad       Padding type to use for additional padded dimensions
                ///
                AvgPool(const Output<Node>& arg,
                        const Strides& strides,
                        const Shape& pads_begin,
                        const Shape& pads_end,
                        const Shape& kernel,
                        bool exclude_pad,
                        op::RoundingType rounding_type,
                        const PadType& auto_pad);

                ///
                /// \brief      Constructs a batched average pooling operation.
                ///
                /// \param      arg            The output producing the input data batch tensor.<br>
                ///                            `[d1, dn]`
                /// \param      strides        The strides.<br> `[n]`
                /// \param      pads_begin     The beginning of padding shape.<br> `[n]`
                /// \param      pads_end       The end of padding shape.<br> `[n]`
                /// \param      kernel         The kernel shape.<br> `[n]`
                /// \param      exclude_pad    If false then averages include padding elements, each
                ///                            treated as the number zero.  If true, padding
                ///                            elements
                ///                            are entirely ignored when computing averages.
                /// \param      rounding_type  Whether to use ceiling or floor rounding type while
                ///                            computing output shape.
                ///
                AvgPool(const Output<Node>& arg,
                        const Strides& strides,
                        const Shape& pads_begin,
                        const Shape& pads_end,
                        const Shape& kernel,
                        bool exclude_pad,
                        op::RoundingType rounding_type);

                size_t get_version() const override { return 1; }
                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;

                /// \return The kernel shape.
                const Shape& get_kernel() const;
                void set_kernel(const Shape& kernel);
                /// \return The strides.
                const Strides& get_strides() const;
                void set_strides(const Strides& strides);
                /// \return The beginning of padding shape.
                const Shape& get_pads_begin() const;
                void set_pads_begin(const Shape& pads_begin);
                /// \return The end of padding shape.
                const Shape& get_pads_end() const;
                void set_pads_end(const Shape& pads_end);
                bool get_exclude_pad() const;
                void set_exclude_pad(bool exclude_pad);
                /// \return The pad type for pooling.
                const PadType& get_auto_pad() const;
                void set_auto_pad(const PadType& auto_pad);
                op::RoundingType get_rounding_type() const;
                void set_rounding_type(op::RoundingType rounding_type);
                /// \return The default value for AvgPool.
                virtual std::shared_ptr<Node> get_default_value() const override;

            protected:
                Shape m_kernel;
                Strides m_strides;
                Shape m_pads_begin;
                Shape m_pads_end;
                bool m_exclude_pad{true};
                PadType m_auto_pad{PadType::EXPLICIT};
                op::RoundingType m_rounding_type{op::RoundingType::FLOOR};
            };

            class NGRAPH_API AvgPoolBackprop : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"AvgPoolBackprop", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                AvgPoolBackprop() = default;
                AvgPoolBackprop(const Output<Node>& delta,
                                const Output<Node>& forward_arg_shape,
                                const Strides& strides,
                                const Shape& pads_begin,
                                const Shape& pads_end,
                                const Shape& kernel,
                                bool exclude_pad);

                size_t get_version() const override { return 1; }
                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                const Shape get_forward_arg_shape() const;
                const Shape& get_kernel() const;
                void set_kernel(const Shape& kernel);
                const Strides& get_strides() const;
                void set_strides(const Strides& strides);
                const Shape& get_pads_begin() const;
                void set_pads_begin(const Shape& pads_begin);
                const Shape& get_pads_end() const;
                void set_pads_end(const Shape& padding_abve);
                bool get_exclude_pad() const;
                void set_exclude_pad(bool exclude_pad);

            protected:
                Shape m_kernel;
                Strides m_strides;
                Shape m_pads_begin;
                Shape m_pads_end;
                bool m_exclude_pad{false};
            };
        } // namespace v1

        using v0::AvgPool;
        using v0::AvgPoolBackprop;
    } // namespace op
} // namespace ngraph
