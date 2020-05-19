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

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief CompiledKernel represents a sub-graph that can be compiled and executed
            /// independently.
            ///
            /// This op can be used to delimit sub-graphs that with special compilation requirements
            /// within a function. For example, we currently use it to delimit sub-graphs that will
            /// be independently compiled and executed by MLIR backend.
            class NGRAPH_API CompiledKernel : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"CompiledKernel", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                CompiledKernel() = default;
                CompiledKernel(const NodeVector& node_list,
                               const OutputVector& outputs,
                               const OutputVector& args);
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                const NodeVector& get_node_list() const { return m_node_list; }
                const OutputVector& get_kernel_outputs() const { return m_outputs; }
                // For node B inside CompiledKernel ck such that A->B and A is outside of ck:
                // replace input to B with a dummy Parameter Op and add an entry to ck's
                // m_input_map.
                void encapsulate_nodes();
                const std::unordered_map<std::shared_ptr<Node>, size_t>& get_input_map() const
                {
                    return m_input_map;
                }
                void insert_to_input_map(std::shared_ptr<Node>, size_t);

            private:
                NodeVector m_node_list;
                OutputVector m_outputs;
                // Used to store the information of internal nodes that have input coming from
                // outside of CK
                std::unordered_map<std::shared_ptr<Node>, size_t> m_input_map;
            };
        }
        using v0::CompiledKernel;
    }
}
