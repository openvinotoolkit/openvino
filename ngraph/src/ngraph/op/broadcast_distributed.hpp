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

#include <memory>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            class NGRAPH_API BroadcastDistributed : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"BroadcastDistributed", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                BroadcastDistributed() = default;
                BroadcastDistributed(const Output<Node>& arg, int64_t root_id = 0);
                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                int64_t get_root_id() const;
                void set_root_id(int64_t root_id);

            private:
                int64_t m_root_id;
            };
        }
        using v0::BroadcastDistributed;
    }
}
