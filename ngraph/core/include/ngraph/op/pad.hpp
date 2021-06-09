// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Generic padding operation.
            class NGRAPH_API Pad : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Pad", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a generic padding operation.
                ///
                /// \param arg The output producing input tensor to be padded.
                /// \param pads_begin The output which specifies the number of padding elements
                /// added
                /// before position 0 on each axis of arg.
                /// \param pads_end The output which specifies the number of padding elements
                /// after the last element on each axis.
                /// \param arg_pad_value The scalar output with the value used for padding
                /// if pad_mode is CONSTANT
                /// \param pad_mode The padding mode: CONSTANT, EDGE, REFLECT or SYMMETRIC.
                /// CONSTANT initializes new elements with arg_pad_value, EDGE uses the nearest
                /// value from arg. REFLECT and SYMMETRIC tile the background by flipping arg
                /// at the edge (SYMMETRIC) or on the last row/column/etc. (REFLECT).
                Pad(const Output<Node>& arg,
                    const Output<Node>& pads_begin,
                    const Output<Node>& pads_end,
                    const Output<Node>& arg_pad_value,
                    PadMode pad_mode);

                /// \brief Constructs a generic padding operation.
                ///
                /// \param arg The output producing input tensor to be padded.
                /// \param pads_begin The output which specifies the number of padding elements
                /// added
                /// \param pads_end The output which specifies the number of padding elements
                /// after the last element on each axis.
                /// \param pad_mode The padding mode: CONSTANT, EDGE, REFLECT or SYMMETRIC.
                Pad(const Output<Node>& arg,
                    const Output<Node>& pads_begin,
                    const Output<Node>& pads_end,
                    PadMode pad_mode);

                /// \brief Constructs a generic padding operation.
                Pad() = default;

                bool visit_attributes(AttributeVisitor& visitor) override;
                size_t get_version() const override { return 1; }
                void validate_and_infer_types() override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// return The node which specifies the number of padding elements
                /// added at the beginning of each axis
                CoordinateDiff get_pads_begin() const;
                /// return The node which specifies the number of padding elements
                /// added at the end of each axis
                CoordinateDiff get_pads_end() const;

                /// \return The padding mode.
                PadMode get_pad_mode() const { return m_pad_mode; }
                void set_pad_mode(PadMode pad_mode) { m_pad_mode = pad_mode; }
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;

            private:
                PadMode m_pad_mode;
                bool evaluate_pad(const HostTensorVector& outputs,
                                  const HostTensorVector& inputs) const;
            };
        } // namespace v1
    }     // namespace op
} // namespace ngraph
