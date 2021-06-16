// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Batched max pooling operation.
            class NGRAPH_API MaxPool : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

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
                        op::RoundingType rounding_mode = op::RoundingType::FLOOR,
                        const PadType& auto_pad = op::PadType::EXPLICIT);

                bool visit_attributes(AttributeVisitor& visitor) override;
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
                NGRAPH_SUPPRESS_DEPRECATED_START
                virtual std::shared_ptr<Node> get_default_value() const override;
                NGRAPH_SUPPRESS_DEPRECATED_END

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;

            protected:
                Shape m_kernel;
                Strides m_strides;
                Shape m_pads_begin;
                Shape m_pads_end;
                PadType m_auto_pad;
                op::RoundingType m_rounding_type;

            private:
                bool update_auto_padding(const PartialShape& in_shape,
                                         Shape& new_pads_end,
                                         Shape& new_pads_begin) const;
                bool evaluate_maxpool(const HostTensorVector& outputs,
                                      const HostTensorVector& inputs) const;
            };
        } // namespace v1

        namespace v8
        {
            /// \brief MaxPooling operation with values and indices calculated as individual outputs
            class NGRAPH_API MaxPool : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Constructs an empty MaxPool operation.
                MaxPool() = default;

                /// \brief Constructs a parametrized MaxPool operation.
                ///
                /// \param arg Output of a node producing the feature tensor to be pooled.
                /// \param strides The strides of the pooling filter.
                /// \param dilations The dilations of the pooling filter.
                /// \param pads_begin Paddings at the beginning of each spatial axis.
                /// \param pads_end Paddings at the end of each spatial axis.
                /// \param kernel The kernel shape.
                /// \param rounding_mode Whether to use ceiling or floor rounding type while
                ///                      computing the output shape.
                /// \param auto_pad The pad type for automatic calculation of the padding sizes.
                /// \param index_element_type The datatype used by the second output tensor
                ///                           containing the selected indices.
                MaxPool(const Output<Node>& arg,
                        const Strides& strides,
                        const Strides& dilations,
                        const Shape& pads_begin,
                        const Shape& pads_end,
                        const Shape& kernel,
                        const op::RoundingType rounding_mode = op::RoundingType::FLOOR,
                        const PadType& auto_pad = op::PadType::EXPLICIT,
                        const element::Type_t index_element_type = element::i32);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The kernel shape.
                const Shape& get_kernel() const { return m_kernel; }
                void set_kernel(const Shape& kernel) { m_kernel = kernel; }
                /// \return The strides.
                const Strides& get_strides() const { return m_strides; }
                void set_strides(const Strides& strides) { m_strides = strides; }
                /// \return The dilations.
                const Strides& get_dilations() const { return m_dilations; }
                void set_dilations(const Strides& dilations) { m_dilations = dilations; }
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

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;

            protected:
                Strides m_strides;
                Strides m_dilations;
                Shape m_pads_begin;
                Shape m_pads_end;
                Shape m_kernel;
                op::RoundingType m_rounding_type;
                PadType m_auto_pad;
                element::Type_t m_index_element_type;

            private:
                // bool update_auto_padding(const PartialShape& in_shape,
                //                          Shape& new_pads_end,
                //                          Shape& new_pads_begin) const;
                bool evaluate_maxpool(const HostTensorVector& outputs,
                                      const HostTensorVector& inputs) const;
            };
        } // namespace v8
    }     // namespace op
} // namespace ngraph
