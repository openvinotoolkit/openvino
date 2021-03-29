// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            class NGRAPH_API Result : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Result", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Allows a value to be used as a function result.
                Result() = default;
                /// \brief Allows a value to be used as a function result.
                ///
                /// \param arg Node that produces the input tensor.
                Result(const Output<Node>& arg, bool needs_default_layout = false);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                void set_needs_default_layout(bool val) { m_needs_default_layout = val; }
                bool needs_default_layout() const { return m_needs_default_layout; }
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool constant_fold(OutputVector& output_values,
                                   const OutputVector& inputs_values) override;

            private:
                bool m_needs_default_layout{false};
            };
        }

        using v0::Result;
    }
    using ResultVector = std::vector<std::shared_ptr<op::Result>>;

    template <>
    class NGRAPH_API AttributeAdapter<ResultVector> : public VisitorAdapter
    {
    public:
        AttributeAdapter(ResultVector& ref);

        bool visit_attributes(AttributeVisitor& visitor) override;

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<ResultVector>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }

    protected:
        ResultVector& m_ref;
    };
}
