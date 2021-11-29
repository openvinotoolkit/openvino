// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            class NGRAPH_API Reverse : public Op
            {
            public:
                enum class Mode
                {
                    INDEX,
                    MASK
                };

                static constexpr NodeTypeInfo type_info{"Reverse", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Reverse() = default;
                /// \brief Constructs a reverse operation.
                ///
                /// \param data The input tensor, some of whose axes are to be reversed.
                /// \param reversed_axes The axes to reverse in a form of a set of indices or
                /// boolean mask.
                /// \param mode The way reversed_axes should be interpreted - a set or a mask.
                Reverse(const Output<Node>& data,
                        const Output<Node>& reversed_axes,
                        const std::string& mode);

                Reverse(const Output<Node>& data,
                        const Output<Node>& reversed_axes,
                        const Mode mode);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The second input data interpretation mode.
                Mode get_mode() const { return m_mode; }
                void set_mode(const Mode mode) { m_mode = mode; }
                virtual size_t get_version() const override { return 1; }
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;

            protected:
                Mode mode_from_string(const std::string& mode) const;

                /// \brief Indicates how the values from the second input should be interpreted.
                ///
                /// The second input can contain a set of indices pointing to axes in the data
                /// tensor shape.
                /// Alternatively it can contain a boolean mask that indicates which axes should be
                /// reversed.
                Mode m_mode;

            private:
                bool evaluate_reverse(const HostTensorVector& outputs,
                                      const HostTensorVector& inputs) const;
            };
        } // namespace v1
    }     // namespace op

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const op::v1::Reverse::Mode& type);

    template <>
    class NGRAPH_API AttributeAdapter<op::v1::Reverse::Mode>
        : public EnumAttributeAdapterBase<op::v1::Reverse::Mode>
    {
    public:
        AttributeAdapter(op::v1::Reverse::Mode& value)
            : EnumAttributeAdapterBase<op::v1::Reverse::Mode>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<op::v1::Reverse::Mode>", 1};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
} // namespace ngraph
