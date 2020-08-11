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

#include <string>
#include <tuple>
#include <vector>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// An op directly representing backend-specific code.
            ///
            /// N.B. Not all backends support all operation languages; a
            /// given backend might only support a given passthrough
            /// operation language in certain modes.
            class Passthrough;
        }
        using v0::Passthrough;
    }
}

class NGRAPH_API ngraph::op::v0::Passthrough final : public Op
{
public:
    static constexpr NodeTypeInfo type_info{"Passthrough", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }
    Passthrough() = default;

    Passthrough(const std::string& logical_type, // aka "What this operation is doing"
                const std::string& language,     // The language the implementation is written in
                const std::string& function,     // The operation implementation
                const OutputVector& args,
                std::vector<std::tuple<element::Type, PartialShape>> outputs);

    void validate_and_infer_types() final;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const final;

    const std::string& logical_type() const { return m_logical_type; }
    const std::string& language() const { return m_language; }
    const std::string& function() const { return m_function; }
    const std::vector<std::tuple<element::Type, PartialShape>>& output_shapes() const
    {
        return m_output_shapes;
    }

private:
    std::string m_logical_type;
    std::string m_language;
    std::string m_function;
    std::vector<std::tuple<element::Type, PartialShape>> m_output_shapes;
};
