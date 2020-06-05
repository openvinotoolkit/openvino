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
            /// \brief Batched max pooling operation, with optional padding and window stride.
            class NGRAPH_API MaxPool : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"MaxPool", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a batched max pooling operation.
                MaxPool() = default;

                /// \brief Constructs a batched max pooling operation.
                ///
                /// \param arg The node producing the input data batch tensor.
                /// \param window_shape The window shape.
                /// \param window_movement_strides The window movement strides.
                /// \param padding_below The below-padding shape.
                /// \param padding_above The above-padding shape.
                /// \param pad_type The pad type for automatically computing padding sizes
                /// \param ceil_mode Whether to use ceiling while computing output shape.
                MaxPool(const Output<Node>& arg,
                        const Shape& window_shape,
                        const Strides& window_movement_strides,
                        const Shape& padding_below,
                        const Shape& padding_above,
                        const PadType& pad_type,
                        bool ceil_mode);

                /// \brief Constructs a batched max pooling operation.
                ///
                /// \param arg The node producing the input data batch tensor.
                /// \param window_shape The window shape.
                /// \param window_movement_strides The window movement strides.
                /// \param padding_below The below-padding shape.
                /// \param padding_above The above-padding shape.
                /// \param pad_type The pad type for automatically computing padding sizes
                MaxPool(const Output<Node>& arg,
                        const Shape& window_shape,
                        const Strides& window_movement_strides,
                        const Shape& padding_below,
                        const Shape& padding_above,
                        const PadType& pad_type);

                /// \brief Constructs a batched max pooling operation.
                ///
                /// \param arg The node producing the input data batch tensor.
                /// \param window_shape The window shape.
                /// \param window_movement_strides The window movement strides.
                /// \param padding_below The below-padding shape.
                /// \param padding_above The above-padding shape.
                MaxPool(const Output<Node>& arg,
                        const Shape& window_shape,
                        const Strides& window_movement_strides,
                        const Shape& padding_below,
                        const Shape& padding_above);

                void validate_and_infer_types() override;

                /// \brief Constructs a batched, unpadded max pooling operation (i.e., all padding
                ///        shapes are set to 0).
                ///
                /// \param arg The node producing the input data batch tensor.
                /// \param window_shape The window shape.
                /// \param window_movement_strides The window movement strides.
                MaxPool(const Output<Node>& arg,
                        const Shape& window_shape,
                        const Strides& window_movement_strides);

                /// \brief Constructs an unstrided batched max pooling operation (i.e., all window
                ///        movement strides are 1 and all padding shapes are set to 0).
                ///
                /// \param arg The node producing the input data batch tensor.
                /// \param window_shape The window shape.
                MaxPool(const Output<Node>& arg, const Shape& window_shape);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The window shape.
                const Shape& get_window_shape() const { return m_window_shape; }
                void set_window_shape(const Shape& window_shape) { m_window_shape = window_shape; }
                /// \return The window movement strides.
                const Strides& get_window_movement_strides() const
                {
                    return m_window_movement_strides;
                }
                void set_window_movement_strides(const Strides& window_movement_strides)
                {
                    m_window_movement_strides = window_movement_strides;
                }
                /// \return The below-padding shape.
                const Shape& get_padding_below() const { return m_padding_below; }
                void set_padding_below(const Shape& padding_below)
                {
                    m_padding_below = padding_below;
                }
                /// \return The above-padding shape.
                const Shape& get_padding_above() const { return m_padding_above; }
                void set_adding_above(const Shape& padding_above)
                {
                    m_padding_above = padding_above;
                }
                /// \return The pad type for pooling.
                const PadType& get_pad_type() const { return m_pad_type; }
                void set_pad_type(const PadType& pad_type) { m_pad_type = pad_type; }
                /// \return The ceiling mode being used for output shape computations
                bool get_ceil_mode() const { return m_ceil_mode; }
                void set_ceil_mode(bool ceil_mode) { m_ceil_mode = ceil_mode; }
                /// \return The default value for MaxPool.
                virtual std::shared_ptr<Node> get_default_value() const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;

                Shape m_window_shape;
                Strides m_window_movement_strides;
                Shape m_padding_below;
                Shape m_padding_above;
                PadType m_pad_type;
                bool m_ceil_mode{false};

            private:
                void update_auto_padding(const PartialShape& in_shape,
                                         Shape& new_padding_above,
                                         Shape& new_padding_below);
            };

            class NGRAPH_API MaxPoolBackprop : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"MaxPoolBackprop", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                MaxPoolBackprop() = default;

                MaxPoolBackprop(const Output<Node>& arg_forward,
                                const Output<Node>& delta,
                                const Shape& window_shape,
                                const Strides& window_movement_strides,
                                const Shape& padding_below,
                                const Shape& padding_above);

                MaxPoolBackprop(const Output<Node>& arg_forward,
                                const Output<Node>& delta,
                                const Output<Node>& result_forward,
                                const Shape& window_shape,
                                const Strides& window_movement_strides,
                                const Shape& padding_below,
                                const Shape& padding_above);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                void validate_and_infer_types() override;

                const Shape& get_window_shape() const { return m_window_shape; }
                void set_window_shape(const Shape& window_shape) { m_window_shape = window_shape; }
                const Strides& get_window_movement_strides() const
                {
                    return m_window_movement_strides;
                }
                void set_window_movement_strides(const Strides& window_movement_strides)
                {
                    m_window_movement_strides = window_movement_strides;
                }
                const Shape& get_padding_below() const { return m_padding_below; }
                void set_padding_below(const Shape& padding_below)
                {
                    m_padding_below = padding_below;
                }
                const Shape& get_padding_above() const { return m_padding_above; }
                void set_padding_above(const Shape& padding_above)
                {
                    m_padding_above = padding_above;
                }

            protected:
                Shape m_window_shape;
                Strides m_window_movement_strides;
                Shape m_padding_below;
                Shape m_padding_above;
            };
        } // namespace v0

        namespace v1
        {
            /// \brief Batched max pooling operation.
            class NGRAPH_API MaxPool : public Op
            {
            public:
                RTTI_DECLARATION;
                //static constexpr NodeTypeInfo type_info{"MaxPool", 1};
                //const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a batched max pooling operation.
                MaxPool() = default;

                /// \brief Constructs a batched max pooling operation.
                ///
                /// \param arg The node producing the input data batch tensor.
                /// \param strides The strides.
                /// \param pads_begin The beginning of padding shape.
                /// \param pads_end The end of padding shape.
                /// \param kernel The kernel shape.
                /// \param rounding_mode Whether to use ceiling or floor rounding type while
                /// computing output shape.
                /// \param auto_pad The pad type for automatically computing padding sizes.
                MaxPool(const Output<Node>& arg,
                        const Strides& strides,
                        const Shape& pads_begin,
                        const Shape& pads_end,
                        const Shape& kernel,
                        op::RoundingType rounding_mode,
                        const PadType& auto_pad);

                /// \brief Constructs a batched max pooling operation.
                ///
                /// \param arg The node producing the input data batch tensor.
                /// \param strides The strides.
                /// \param pads_begin The beginning of padding shape.
                /// \param pads_end The end of padding shape.
                /// \param kernel The kernel shape.
                /// \param rounding_mode Whether to use ceiling or floor rounding type while
                /// computing output shape.
                MaxPool(const Output<Node>& arg,
                        const Strides& strides,
                        const Shape& pads_begin,
                        const Shape& pads_end,
                        const Shape& kernel,
                        op::RoundingType rounding_mode);

                bool visit_attributes(AttributeVisitor& visitor) override;
                size_t get_version() const override { return 1; }
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The kernel shape.
                const Shape& get_kernel() const { return m_kernel; }
                void set_kernel(const Shape& kernel) { m_kernel = kernel; }
                /// \return The strides.
                const Strides& get_strides() const { return m_strides; }
                void set_strides(const Strides& strides) { m_strides = strides; }
                /// \return The beginning of padding shape.
                const Shape& get_pads_begin() const { return m_pads_begin; }
                void set_pads_begin(const Shape& pads_begin) { m_pads_begin = pads_begin; }
                /// \return The end of padding shape.
                const Shape& get_pads_end() const { return m_pads_end; }
                void set_adding_above(const Shape& pads_end) { m_pads_end = pads_end; }
                /// \return The pad type for pooling.
                const PadType& get_auto_pad() const { return m_auto_pad; }
                void set_auto_pad(const PadType& auto_pad) { m_auto_pad = auto_pad; }
                /// \return The ceiling mode being used for output shape computations
                op::RoundingType get_rounding_type() const { return m_rounding_type; }
                void set_rounding_type(op::RoundingType rounding_mode)
                {
                    m_rounding_type = rounding_mode;
                }
                /// \return The default value for MaxPool.
                virtual std::shared_ptr<Node> get_default_value() const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;

                Shape m_kernel;
                Strides m_strides;
                Shape m_pads_begin;
                Shape m_pads_end;
                PadType m_auto_pad;
                op::RoundingType m_rounding_type{op::RoundingType::FLOOR};

            private:
                void update_auto_padding(const PartialShape& in_shape,
                                         Shape& new_pads_end,
                                         Shape& new_pads_begin);
            };

            class NGRAPH_API MaxPoolBackprop : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"MaxPoolBackprop", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                MaxPoolBackprop() = default;

                MaxPoolBackprop(const Output<Node>& arg_forward,
                                const Output<Node>& delta,
                                const Strides& strides,
                                const Shape& pads_begin,
                                const Shape& pads_end,
                                const Shape& kernel);

                MaxPoolBackprop(const Output<Node>& arg_forward,
                                const Output<Node>& delta,
                                const Output<Node>& result_forward,
                                const Strides& strides,
                                const Shape& pads_begin,
                                const Shape& pads_end,
                                const Shape& kernel);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                size_t get_version() const override { return 1; }
                void validate_and_infer_types() override;

                const Shape& get_kernel() const { return m_kernel; }
                void set_kernel(const Shape& kernel) { m_kernel = kernel; }
                const Strides& get_strides() const { return m_strides; }
                void set_strides(const Strides& strides) { m_strides = strides; }
                const Shape& get_pads_begin() const { return m_pads_begin; }
                void set_pads_begin(const Shape& pads_begin) { m_pads_begin = pads_begin; }
                const Shape& get_pads_end() const { return m_pads_end; }
                void set_pads_end(const Shape& pads_end) { m_pads_end = pads_end; }
            protected:
                Shape m_kernel;
                Strides m_strides;
                Shape m_pads_begin;
                Shape m_pads_end;
            };
        } // namespace v1

        using v0::MaxPool;
        using v0::MaxPoolBackprop;
    } // namespace op
} // namespace ngraph
