// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v5
        {
            class NGRAPH_API LogSoftmax : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                LogSoftmax() = default;
                /// \brief Constructs a LogSoftmax operation.
                ///
                /// \param arg Node that produces the first input tensor.<br>
                /// `[d0, ...]`
                /// \param axis The axis position (0-based) on which to calculate the LogSoftmax.
                ///
                /// Output `[d0, ...]`
                ///
                LogSoftmax(const Output<Node>& arg, const int64_t axis);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                size_t get_version() const override { return 1; }
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                int64_t get_axis() const { return m_axis; }
                void set_axis(const int64_t axis) { m_axis = axis; }

            private:
                int64_t m_axis = 1;
            };
        } // namespace v5
    }     // namespace op
} // namespace ngraph
