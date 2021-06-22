// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/runtime/host_tensor.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Dynamic Tiling operation which repeats a tensor multiple times
            ///        along each dimension
            class NGRAPH_API Tile : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Tile", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Tile() = default;
                /// \brief Perform dynamic padding of a tensor
                ///
                /// \param data The node producing input tensor to be padded.
                /// \param repeats The node producing the per-dimension replication factor
                Tile(const Output<Node>& data, const Output<Node>& repeats);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;

            private:
                bool evaluate_tile(const HostTensorVector& outputs,
                                   const HostTensorVector& inputs) const;
            };
        } // namespace v0
    }     // namespace op
} // namespace ngraph
