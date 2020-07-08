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

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            ///
            /// \brief      Class performing element-wise linear quantization.
            ///
            /// \note       Input floating point values are quantized into a discrete
            ///             set of floating point values.
            ///
            /// \paragraph Implementation This class creates a node which performs the following
            ///            operation:
            ///
            ///            round((data - input_low) / (input_high - input_low) * (levels-1)) /
            ///                 (levels-1) * (output_high - output_low) + output_low
            ///
            ///
            class NGRAPH_API FakeQuantize : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"FakeQuantize", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                FakeQuantize() = default;
                ///
                /// \brief      Constructs a FakeQuantize operation node.
                ///
                /// \param[in]  data            The input data tensor.
                /// \param[in]  input_low       The minimum limit for input values.
                /// \param[in]  input_high      The maximum limit for input values.
                /// \param[in]  output_low      The minimum quantized value.
                /// \param[in]  output_high     The maximum quantized value.
                /// \param[in]  levels          The number of quantization levels.
                /// \param[in]  auto_broadcast  AutoBroadcast mode to be used for broadcasting
                ///                             limit values
                ///
                FakeQuantize(const Output<Node>& data,
                             const Output<Node>& input_low,
                             const Output<Node>& input_high,
                             const Output<Node>& output_low,
                             const Output<Node>& output_high,
                             std::size_t levels,
                             const AutoBroadcastSpec& auto_broadcast =
                                 AutoBroadcastSpec(AutoBroadcastType::NUMPY));

                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual OutputVector decompose_op() const override;
                virtual void validate_and_infer_types() override;

                // This is a hack to work around dldt directly calling copy_with_new_args
                // When that code is replace with clone_with_new_inputs then remove this
                // method.
                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                std::size_t get_levels() const { return m_levels; }
                void set_levels(std::size_t levels) { m_levels = levels; }
                const AutoBroadcastSpec& get_auto_broadcast() const { return m_auto_broadcast; }
                void set_auto_broadcast(const AutoBroadcastSpec& auto_broadcast)
                {
                    m_auto_broadcast = auto_broadcast;
                }

            private:
                std::size_t m_levels;
                AutoBroadcastSpec m_auto_broadcast;
            };
        }
        using v0::FakeQuantize;
    }
}
